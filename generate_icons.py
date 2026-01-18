from PIL import Image, ImageDraw

def create_icon(size, filename):
    img = Image.new('RGB', (size, size), color='#0ea5e9')
    d = ImageDraw.Draw(img)
    # Draw a cloud-like shape (simplified as circles)
    d.ellipse([size*0.2, size*0.3, size*0.6, size*0.7], fill='#f8fafc')
    d.ellipse([size*0.4, size*0.2, size*0.8, size*0.6], fill='#f8fafc')
    img.save(filename)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_icon(192, "c:/Users/Arfeen/Desktop/CloudChaser/frontend/public/icons/icon-192.png")
    create_icon(512, "c:/Users/Arfeen/Desktop/CloudChaser/frontend/public/icons/icon-512.png")
