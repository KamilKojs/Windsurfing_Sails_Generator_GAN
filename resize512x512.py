import os
import cv2

for root, subdirs, files in os.walk("/Users/kamil/Desktop/Windsurfing_Sails_Photos_resized"):

    for filename in files:
        file_path = os.path.join(root, filename)
        print('\t- file %s (full path: %s)' % (filename, file_path))

        img = cv2.imread(file_path)

        bType = cv2.BORDER_REPLICATE
        img_sq = img.copy()
        (h, w) = img_sq.shape[:2]

        #bColor = [int(item) for item in args.border_color.split(',')]

        (h, w) = img_sq.shape[:2]

        if (h > w):
            # pad left/right
            diff = h - w
            if (diff % 2 == 0):
                img_sq = cv2.copyMakeBorder(img_sq, 0, 0, int(diff / 2), int(diff / 2), bType)
            else:
                img_sq = cv2.copyMakeBorder(img_sq, 0, 0, int(diff / 2) + 1, int(diff / 2), bType)
        elif (w > h):
            # pad top/bottom
            diff = w - h
            if (diff % 2 == 0):
                img_sq = cv2.copyMakeBorder(img_sq, int(diff / 2), int(diff / 2), 0, 0, bType)
            else:
                img_sq = cv2.copyMakeBorder(img_sq, int(diff / 2), int(diff / 2) + 1, 0, 0, bType)
        else:
            diff = scale - h
            if (diff % 2 == 0):
                img_sq = cv2.copyMakeBorder(img_sq, int(diff / 2), int(diff / 2), int(diff / 2), int(diff / 2), bType)
            else:
                img_sq = cv2.copyMakeBorder(img_sq, int(diff / 2), int(diff / 2) + 1, int(diff / 2), int(diff / 2) + 1,
                                            bType)


        new_file = os.path.splitext(filename)[0] + ".png"
        cv2.imwrite(os.path.join("/Users/kamil/Desktop/new", new_file), img_sq, [cv2.IMWRITE_PNG_COMPRESSION, 0])


