import cv2
import os


def get_jpg_files(directory):
    png_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                png_files.append(os.path.join(root, file))
    return png_files


def match_image(cropped_image, database):
    cropped = cv2.imread(cropped_image)
    cv2.imwrite('cropping.jpg', cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cropped = cv2.imread('cropping.jpg')
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    best_match = {"filename": None, "similarity": 0.0}


    for filename in database:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.xfeatures2d.SIFT_create()
        keypoint, descriptor1 = sift.detectAndCompute(cropped, None)

        _, descriptor2 = sift.detectAndCompute(image, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptor1, descriptor2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        similarity = len(good) / len(keypoint)
        
        if similarity > best_match["similarity"]:
            best_match = {"filename": filename, "similarity": similarity}

    return best_match["filename"]


if __name__ == "__main__":
    all_images = get_jpg_files("a")
    matched = match_image("image.png", all_images)
    """
    table = {45: 32}
    print(matched)
    print()
    print(matched[2:-10].translate(table))
    print()
    """