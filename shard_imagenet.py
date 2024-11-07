import os
import tarfile
import glob
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Shard ImageNet dataset into WebDataset-compatible tar files.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to extracted ImageNet images with class subdirectories.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save sharded tar files.")
    parser.add_argument('--num_images_per_shard', type=int, default=1000, help="Number of images per shard.")
    parser.add_argument('--label_mapping_file', type=str, required=True, help="Path to label mapping file with class-to-integer mappings.")
    return parser.parse_args()

def load_label_mapping(label_mapping_file):
    label_mapping = {}
    with open(label_mapping_file, "r") as f:
        for line in f:
            class_name, label_id = line.strip().split()
            label_mapping[class_name] = int(label_id)
    return label_mapping

def create_shards(input_dir, output_dir, num_images_per_shard, label_mapping):
    os.makedirs(output_dir, exist_ok=True)
    all_images = glob.glob(f'{input_dir}/*/*.JPEG')
    random.shuffle(all_images)  # Shuffle the list of images
    print(f"Found {len(all_images)} images.")  # Debug: check number of images detected

    # Split images into chunks manually
    for i in range(0, len(all_images), num_images_per_shard):
        batch = all_images[i:i + num_images_per_shard]
        print(f"Creating shard {i // num_images_per_shard} with {len(batch)} images.")  # Debug: check batch sizes
        # print(batch)
        shard_filename = os.path.join(output_dir, f'imagenet-{i // num_images_per_shard:04d}.tar')
        with tarfile.open(shard_filename, 'w') as tar:
            for filepath in batch:
                image_name = os.path.splitext(os.path.basename(filepath))[0]
                class_name = os.path.basename(os.path.dirname(filepath))  # Class name from directory name
                class_id = label_mapping.get(class_name, -1)  # Get the integer label, -1 if not found
                
                # Add image file directly to tar
                tar.add(filepath, arcname=f"{image_name}.png")
                
                # Create and add class file
                class_file = f"{image_name}.cls"
                with open(class_file, "w") as f:
                    f.write(str(class_id))
                tar.add(class_file, arcname=f"{image_name}.cls")
                
                # Clean up temporary class file
                os.remove(class_file)
        
        print(f"Created shard: {shard_filename}")

if __name__ == "__main__":
    args = parse_args()
    label_mapping = load_label_mapping(args.label_mapping_file)
    create_shards(args.input_dir, args.output_dir, args.num_images_per_shard, label_mapping)




