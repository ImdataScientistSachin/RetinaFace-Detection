import argparse
import cv2
import os
from src.detector import FaceDetector

def main():
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Ratina_Face: Industry-grade Face Detection & Verification")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Detect Subcommand
    detect_parser = subparsers.add_parser("detect", help="Run face detection on images")
    detect_parser.add_argument("--image", type=str, help="Path to a single image file")
    detect_parser.add_argument("--input_dir", type=str, help="Path to a directory of images")
    detect_parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    detect_parser.add_argument("--threshold", type=float, default=0.9, help="Detection threshold")
    
    # Verify Subcommand
    verify_parser = subparsers.add_parser("verify", help="Verify if two images are of the same person")
    verify_parser.add_argument("img1", type=str, help="Path to the first image")
    verify_parser.add_argument("img2", type=str, help="Path to the second image")
    verify_parser.add_argument("--model", type=str, default="ArcFace", help="Verification model (e.g. ArcFace, Facenet)")

    args = parser.parse_args()

    # Initialize detector
    detector = FaceDetector(threshold=getattr(args, 'threshold', 0.9))
    if hasattr(args, 'model'):
        detector.verification_model = args.model

    if args.command == "detect":
        # Ensure output directory exists
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        if args.image:
            process_image(args.image, detector, args.output_dir)
        elif args.input_dir:
            for filename in os.listdir(args.input_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(args.input_dir, filename)
                    process_image(image_path, detector, args.output_dir)
        else:
            detect_parser.print_help()
            
    elif args.command == "verify":
        print(f"[*] Verifying identity: {args.img1} vs {args.img2}")
        try:
            # The verify method now uses retinaface backend by default in detector.py
            result = detector.verify(args.img1, args.img2)
            print("-" * 40)
            status = "MATCH VERIFIED" if result["verified"] else "NO MATCH"
            print(f"[{'+' if result['verified'] else '-'}] Result: {status}")
            print(f"[*] Distance: {result['distance']:.4f}")
            print(f"[*] Threshold: {result['threshold']:.4f}")
            print(f"[*] Model: {result['model']}")
            print("-" * 40)
        except Exception as e:
            print(f"[!] Error during verification: {e}")
    else:
        parser.print_help()

def process_image(path, detector, output_dir):
    """Helper to process a single image and save the result."""
    print(f"[*] Processing: {path}")
    img = cv2.imread(path)
    if img is None:
        print(f"[!] Error: Could not read image {path}")
        return

    results = detector.detect(img)
    annotated_img = detector.draw_results(img, results)
    output_path = os.path.join(output_dir, f"detected_{os.path.basename(path)}")
    cv2.imwrite(output_path, annotated_img)
    print(f"[+] Saved result to: {output_path} (Found {len(results)} faces)")

if __name__ == "__main__":
    main()
