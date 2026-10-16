import json
import shutil
from pathlib import Path

import typer
from PIL import Image
from rich.console import Console
from rich.progress import track

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
DEFAULT_JSON_TEMPLATE = {
    "version": "6.3.1",
    "flags": {},
    "shapes": [],
    "imagePath": None,
    "imageData": None,
    "imageHeight": None,
    "imageWidth": None,
}

console = Console()


def create_output_directory(output_path, source_path, folder_name) -> Path:
    output_path = output_path or source_path.resolve().parent / folder_name
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def to_pixel(coord, size):
    """归一化坐标 -> 像素坐标, 保留 6 位小数。

    与 YOLO 6 位归一化精度对应: 6 位小数像素再归一化仍还原原值, 往返无损,
    同时避免 0.55*100=55.00000000000001 这类浮点尾数噪声。
    """
    return round(coord * size, 6)


cli = typer.Typer(rich_markup_mode="rich", help="YOLO 标签转 LabelMe 标签 (关键点检测)")


def parse_pose_line(line):
    """解析一行 YOLO pose 标签: class_id cx cy w h x1 y1 v1 x2 y2 v2 ...

    框为归一化中心点+宽高, 每个关键点为归一化坐标 + 可见性(0=缺失, 1=遮挡, 2=可见)。
    返回 ((class_id, box_cxcywh, [(kx, ky, v), ...]), None) 或 (None, 错误信息)。
    """
    parts = line.split()
    if len(parts) < 5:
        return None, f"字段数不足({len(parts)}/5)"
    try:
        class_id = int(parts[0])
        box = [float(v) for v in parts[1:5]]
        kpt_values = [float(v) for v in parts[5:]]
    except ValueError as e:
        return None, f"数值解析失败: {e}"
    if len(kpt_values) % 3 != 0:
        return None, f"关键点字段数不是 3 的倍数({len(kpt_values)})"
    keypoints = [
        (kpt_values[i], kpt_values[i + 1], int(kpt_values[i + 2]))
        for i in range(0, len(kpt_values), 3)
    ]
    return (class_id, box, keypoints), None


def convert_yolo_pose_to_labelme(txt_path, json_path, classes, keypoint_names,
                                 img_width, img_height, image_name, skipped_shapes):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    json_data = DEFAULT_JSON_TEMPLATE.copy()
    json_data.update(
        {
            "imagePath": image_name,
            "imageHeight": img_height,
            "imageWidth": img_width,
            "shapes": [],
        }
    )

    for line_idx, line in enumerate(lines):
        parsed, err = parse_pose_line(line)
        if err is not None:
            skipped_shapes.append((txt_path.name, err, line))
            continue

        class_id, box, keypoints = parsed
        if class_id < 0 or class_id >= len(classes):
            skipped_shapes.append(
                (txt_path.name, f"class_id={class_id} 超出 classes.txt 范围({len(classes)} 类)", line)
            )
            continue

        cx, cy, w, h = box
        x_min = to_pixel(cx - w / 2, img_width)
        y_min = to_pixel(cy - h / 2, img_height)
        x_max = to_pixel(cx + w / 2, img_width)
        y_max = to_pixel(cy + h / 2, img_height)

        group_id = line_idx  # 框与关键点同组, 保证与 labelme_to_yolo_pose.py 分组约定一致

        bbox_shape = {
            "label": classes[class_id],
            "points": [[x_min, y_min], [x_max, y_max]],
            "group_id": group_id,
            "description": None,
            "shape_type": "rectangle",
            "flags": {},
            "mask": None,
        }
        json_data["shapes"].append(bbox_shape)

        for kpt_idx, (kx, ky, v) in enumerate(keypoints):
            if v == 0:
                # 可见性 0 = 未标注, 不生成点 (与 labelme_to_yolo_pose.py 缺失输出 0 0 0 对应)
                continue
            name = (
                keypoint_names[kpt_idx]
                if kpt_idx < len(keypoint_names)
                else f"keypoint_{kpt_idx + 1}"
            )
            json_data["shapes"].append(
                {
                    "label": name,
                    "points": [[to_pixel(kx, img_width), to_pixel(ky, img_height)]],
                    "group_id": group_id,
                    "description": None,
                    "shape_type": "point",
                    "flags": {},
                    "mask": None,
                }
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


@cli.command()
def process_yolo_pose_to_labelme(
    image_path: Path = typer.Argument(..., help="图片目录"),
    class_path: Path = typer.Argument(..., help="classes.txt（每行一个类别名，行号即 class_id）"),
    keypoint_path: Path = typer.Option(
        None,
        "--keypoint_path",
        "-k",
        help="关键点名称文件(每行一个, 顺序即关键点顺序), 不传则自动命名为 keypoint_1...",
    ),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签 txt 目录（默认与图片同目录）"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
):
    """
    将 YOLO pose 标签 (.txt) 转换为 LabelMe 标注 (.json)

    说明:
        1. 遍历图片目录, 每张图片查找同名的 .txt 标签文件
        2. 标签格式: class_id cx cy w h x1 y1 v1 x2 y2 v2 ...
           (框为归一化中心点+宽高, 每个关键点为归一化坐标+可见性)
        3. 每个实例输出一个 rectangle 框 + 若干 point 关键点, 框与关键点
           共享 group_id (从 0 起递增), 与 labelme_assign_group_id.py /
           labelme_to_yolo_pose.py 的分组约定一致
        4. 可见性 v=0 的关键点视为未标注, 不生成点; v=1/2 均生成点
        5. 关键点名称优先取 --keypoint_path 文件中对应顺序的名称,
           不足或未传时自动命名为 keypoint_1, keypoint_2, ...
        6. classes.txt 每行一个类别名, 行号即 class_id, 用于还原类别名
        7. 字段数不足 / 关键点字段数不是 3 的倍数 / class_id 越界 / 数值非法的行跳过并警告
        8. 没有对应 txt 的图片仅拷贝, 不生成 json
        9. 转换结果默认输出到图片目录同级的 yolo2json_pose 文件夹

    使用示例:
        1. 【基本转换】关键点顺序与名称通过文件指定
            python yolo_pose_to_labelme.py ./pose_yolo ./classes.txt -k ./keypoint_names.txt

        2. 【标签目录分离】标签在 labels 目录
            python yolo_pose_to_labelme.py ./images ./classes.txt -k ./keypoint_names.txt -l ./labels

        3. 【自动命名关键点】
            python yolo_pose_to_labelme.py ./images ./classes.txt
    """
    images = [
        f
        for f in image_path.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    label_path = label_path or image_path
    output_path = create_output_directory(output_path, image_path, "yolo2json_pose")

    classes = []
    with open(class_path, "r", encoding="utf-8") as f:
        classes = f.read().splitlines()

    keypoint_names = []
    if keypoint_path is not None:
        with open(keypoint_path, "r", encoding="utf-8") as f:
            keypoint_names = f.read().splitlines()

    skipped_shapes = []
    for img_file in track(images, description="Converting to LabelMe JSON..."):
        img = Image.open(img_file)
        base_name = img_file.stem
        txt_file = label_path / f"{base_name}.txt"
        json_file = output_path / f"{base_name}.json"

        if txt_file.exists():
            convert_yolo_pose_to_labelme(
                txt_file, json_file, classes, keypoint_names,
                img.width, img.height, img_file.name, skipped_shapes
            )
        shutil.copy(img_file, output_path)

    shutil.copy(class_path, output_path / "classes.txt")
    if keypoint_path is not None:
        shutil.copy(keypoint_path, output_path / "keypoint_names.txt")

    for txt_name, reason, line in skipped_shapes:
        console.log(
            f"[yellow]跳过[/yellow] {txt_name}: {reason}\n    {line}"
        )
    console.log(
        f"[green]完成[/green] 共处理 {len(images)} 张图, 生成 {len(images)} 个 json, "
        f"跳过 {len(skipped_shapes)} 条无效标注, 输出目录: {output_path}"
    )


if __name__ == "__main__":
    cli()
