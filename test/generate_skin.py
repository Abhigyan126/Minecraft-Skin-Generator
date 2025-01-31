import torch
from generator import Generatorv3, Generatorv5
import skinpy
import torch
from skinpy import Skin, Perspective
from PIL import Image
import matplotlib.pyplot as plt

#initialise hyperparameter
latent_dim = 128
img_channels = 3

#initialising generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#generator = Generatorv3(latent_dim, img_channels).to(device)
generator = Generatorv5(latent_dim, img_channels).to(device)

#load model and set to evaluation
generator.load_state_dict(torch.load("/Users/abhigyan/Downloads/project/minecraft skin generator/model/model v5.pth", map_location=device))
generator.eval()

def generate_and_render_skin(generator, latent_dim):
    # Generate a single latent vector
    z = torch.randn(1, latent_dim).to(device)

    print(z)
    with torch.no_grad():
        gen_img = generator(z)  # Generate a single image
    
    # Rescale to [0, 1] and move to CPU
    gen_img = (gen_img + 1) / 2
    gen_img = gen_img.cpu().squeeze().permute(1, 2, 0)
    
    # Convert to PIL Image and add alpha channel
    gen_img_pil = Image.fromarray((gen_img.numpy() * 255).astype('uint8'))
    gen_img_rgba = gen_img_pil.convert('RGBA')
    gen_img_rgba.save('test/save/generated_skin.png')
    print("Image generated and saved")
    
    # Create skin and perspective for rendering
    skin = Skin.from_path("test/save/generated_skin.png")
    perspective = Perspective(
        x="left",
        y="front", 
        z="up",
        scaling_factor=10
    )
    
    # Save and render the isometric image
    render = skin.to_isometric_image(perspective)
    render.save("test/save/render.png")
    
    # Display images
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display generated skin
    ax1.imshow(gen_img_rgba)
    ax1.set_title('Generated Skin')
    ax1.axis('off')
    
    # Display rendered skin
    ax2.imshow(render)
    ax2.set_title('Isometric Render')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

# Generate and render the skin
generate_and_render_skin(generator, latent_dim)