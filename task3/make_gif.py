from PIL import Image
import glob

def create_gif(image_paths, output_gif_path, duration=20000):
    images = [Image.open(image_path) for image_path in image_paths]
    # Save as GIF
    images[0].save(
    output_gif_path,
    save_all=True,
    append_images=images[1:],
    duration=duration,
    loop=1 # 0 means infinite loop
 )

if __name__ == "__main__":


    image_list = []
    for filename in glob.glob('rembg_outputs/*.jpg'): 
        image_list.append(filename)

    # Output GIF path
    output_gif_path = "rembg_very_long.gif"
    # Create GIF
    create_gif(image_list, output_gif_path)

print(f"GIF created and saved at {output_gif_path}")