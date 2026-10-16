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


cli = typer.Typer(rich_markup_mode="rich", help="YOLO 标签转 LabelMe 标签 (分割)")


def parse_seg_line(line):
    """解析一行 YOLO seg 标签: class_id x1 y1 x2 y2 ...

    归一化多边形点列。若坐标个数为奇数, 视为末尾带置信度等多余字段, 丢弃最后一项。
    返回 ((class_id, [(x, y), ...]), None) 或 (None, 错误信息)。
    """
    parts = line.split()
    if len(parts) < 7:
        return None, f"字段数不足({len(parts)}, 至少 class + 3 个点)"
    try:
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError as e:
        return None, f"数值解析失败: {e}"
    if len(coords) % 2 != 0:
        coords = coords[:-1]  # 丢弃尾部多余字段(如置信度)
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    if len(points) < 3:
        return None, f"多边形点数不足({len(points)})"
    return (class_id, points), None


def convert_yolo_seg_to_labelme(txt_path, json_path, classes, img_width,
                                img_height, image_name, skipped_shapes):
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

    for line in lines:
        parsed, err = parse_seg_line(line)
        if err is not None:
            skipped_shapes.append((txt_path.name, err, line))
            continue

        class_id, points = parsed
        if class_id < 0 or class_id >= len(classes):
            skipped_shapes.append(
                (txt_path.name, f"class_id={class_id} 超出 classes.txt 范围({len(classes)} 类)", line)
            )
            continue

        json_data["shapes"].append(
            {
                "label": classes[class_id],
                "points": [[to_pixel(x, img_width), to_pixel(y, img_height)] for x, y in points],
                "group_id": None,
                "description": None,
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


@cli.command()
def process_yolo_seg_to_labelme(
    image_path: Path = typer.Argument(..., help="图片目录"),
    class_path: Path = typer.Argument(..., help="classes.txt（每行一个类别名，行号即 class_id）"),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签 txt 目录（默认与图片同目录）"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
):
    """
    将 YOLO seg 标签 (.txt) 转换为 LabelMe 标注 (.json)

    说明:
        1. 遍历图片目录, 每张图片查找同名的 .txt 标签文件
        2. 标签格式: class_id x1 y1 x2 y2 ... (归一化多边形点列)
        3. 归一化坐标按图片宽高还原为像素坐标, shape_type 输出 polygon,
           与 labelme_to_yolo_seg.py 互为可逆转换
        4. 坐标个数为奇数时视为末尾带置信度等多余字段, 自动丢弃最后一项
        5. classes.txt 每行一个类别名, 行号即 class_id, 用于还原类别名
        6. 字段数不足 / 点数不足 3 / class_id 越界 / 数值非法的行跳过并警告
        7. 没有对应 txt 的图片仅拷贝, 不生成 json
        8. 转换结果默认输出到图片目录同级的 yolo2json_seg 文件夹

    使用示例:
        1. 【基本转换】标签与图片在同一目录
            python yolo_seg_to_labelme.py ./seg_yolo ./classes.txt

        2. 【标签目录分离】标签在 labels 目录
            python yolo_seg_to_labelme.py ./images ./classes.txt -l ./labels

        3. 【指定输出目录】
            python yolo_seg_to_labelme.py ./images ./classes.txt -o ./labelme
    """
    images = [
        f
        for f in image_path.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    label_path = label_path or image_path
    output_path = create_output_directory(output_path, image_path, "yolo2json_seg")

    classes = []
    with open(class_path, "r", encoding="utf-8") as f:
        classes = f.read().splitlines()

    skipped_shapes = []
    for img_file in track(images, description="Converting to LabelMe JSON..."):
        img = Image.open(img_file)
        base_name = img_file.stem
        txt_file = label_path / f"{base_name}.txt"
        json_file = output_path / f"{base_name}.json"

        if txt_file.exists():
            convert_yolo_seg_to_labelme(
                txt_file, json_file, classes, img.width, img.height,
                img_file.name, skipped_shapes
            )
        shutil.copy(img_file, output_path)

    shutil.copy(class_path, output_path / "classes.txt")

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
