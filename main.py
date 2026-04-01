import argparse
import os
import sys

import cv2

from src.detector import FaceDetector

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def process_image(path: str, detector: FaceDetector, output_dir: str) -> None:
    print(f"Processing: {path}")
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not read image {path}", file=sys.stderr)
        return

    results = detector.detect(img)
    annotated_img = detector.draw_results(img, results)
    output_path = os.path.join(output_dir, f"detected_{os.path.basename(path)}")
    cv2.imwrite(output_path, annotated_img)
    print(f"Saved: {output_path} (found {len(results)} face(s))")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RetinaFace face detection and identity verification CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    detect_parser = subparsers.add_parser("detect", help="Run face detection on images")
    detect_parser.add_argument("--image", type=str, help="Path to a single image file")
    detect_parser.add_argument(
        "--input_dir", type=str, help="Path to a directory of images"
    )
    detect_parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save results"
    )
    detect_parser.add_argument(
        "--threshold", type=float, default=0.9, help="Detection confidence threshold"
    )

    verify_parser = subparsers.add_parser(
        "verify", help="Verify whether two images show the same person"
    )
    verify_parser.add_argument("img1", type=str, help="Path to the first image")
    verify_parser.add_argument("img2", type=str, help="Path to the second image")
    verify_parser.add_argument(
        "--model",
        type=str,
        default="ArcFace",
        help="Verification model (e.g., ArcFace, Facenet)",
    )

    args = parser.parse_args()

    if args.command == "detect":
        if args.image:
            if not os.path.isfile(args.image):
                print(f"Error: Image file not found: {args.image}", file=sys.stderr)
                return 1
            os.makedirs(args.output_dir, exist_ok=True)
            detector = FaceDetector(threshold=args.threshold)
            process_image(args.image, detector, args.output_dir)
        elif args.input_dir:
            if not os.path.isdir(args.input_dir):
                print(
                    f"Error: Input directory not found: {args.input_dir}",
                    file=sys.stderr,
                )
                return 1
            os.makedirs(args.output_dir, exist_ok=True)
            detector = FaceDetector(threshold=args.threshold)
            for filename in sorted(os.listdir(args.input_dir)):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    process_image(
                        os.path.join(args.input_dir, filename),
                        detector,
                        args.output_dir,
                    )
        else:
            detect_parser.print_help()
            return 1
        return 0

    if args.command == "verify":
        detector = FaceDetector(verification_model=args.model)
        print(f"Verifying identity: {args.img1} vs {args.img2}")
        try:
            result = detector.verify(args.img1, args.img2)
        except Exception as exc:
            print(f"Error during verification: {exc}", file=sys.stderr)
            return 1

        status = "MATCH VERIFIED" if result["verified"] else "NO MATCH"
        print("-" * 40)
        print(f"Result:    {status}")
        print(f"Distance:  {result['distance']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        print(f"Model:     {result['model']}")
        print("-" * 40)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
