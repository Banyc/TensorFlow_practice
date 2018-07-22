from PIL import Image


image1 = Image.open("1.png")
image1_pixels = []
rowPixels = []
for i in range(28):
    for j in range(28):
        pixel = image1.getpixel((j, i))
        pixel = 1.0 - pixel  # 背景色与前景色调换
        rowPixels.append(pixel)
    image1_pixels.append(rowPixels)
    rowPixels = []
print(image1_pixels)
