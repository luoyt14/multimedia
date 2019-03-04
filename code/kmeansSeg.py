import numpy as np
from sklearn.cluster import KMeans
import PIL.Image as image


def loadImage(filename):
    with open(filename, 'rb') as f:
        img = image.open(f)
        m,n = img.size
        data = []

        for i in range(m):
            for j in range(n):
                x, y, z = img.getpixel((i,j))
                data.append([x / 256.0, y / 256.0, z / 256.0])

        return np.mat(data), m, n

def createColor(label):
    color = (label*127, int(255/(label+1)), (2-label)*127)
    return color

if __name__ == '__main__':
    filename = '../images/test9.jpg'
    img, row, col = loadImage(filename)
    label = KMeans(n_clusters=3).fit_predict(img)
    label = label.reshape([row, col])
    pic = image.new('RGBA', (row, col))
    # print(label)

    for i in range(row):
        for j in range(col):
            pic.putpixel((i,j), createColor(label[i][j]))

    pic.show()
    pic.save('../result/test9seg.png')