import os,sys
import xml.etree.ElementTree as ET
import cv2

def convert():
    image_path = sys.argv[1]
    list_path = sys.argv[2]
    save_path = sys.argv[3]
    gt_path = sys.argv[4]
    save_eg = int(sys.argv[5])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    list_file = open(list_path)
    list_lines = list_file.readlines()
    list_file.close()

    ## read in map table from name to label
    map_table_file = open('./name2label.txt')
    lines = map_table_file.readlines()
    map_table_file.close()
    map_table = {}
    index = 0 
    for line in lines:
        line = line.strip('\n')
        line_arr = line.split(' ')
        map_table[line_arr[0]] = line_arr[1]


    for line in list_lines:
        if index % 100 == 0:
            print index
        index += 1

        line = line.strip('\n')
        line_arr = line.split(' ')
        img_id = line_arr[0]
        tree = ET.parse(gt_path + '/' + img_id + '.xml') 

        save_f = open(save_path + '/' + img_id + '.txt', 'w')
        img = cv2.imread(image_path + '/' + img_id + '.jpg')
        
        for obj in tree.findall('object'):
            obj_name = obj.find('name').text
            if not obj_name in map_table:
                print "cannot find %s in map"%(obj_name)
                sys.exit()
            label = map_table[obj_name]
            xmin = obj.find('bndbox').find('xmin').text
            ymin = obj.find('bndbox').find('ymin').text
            xmax = obj.find('bndbox').find('xmax').text
            ymax = obj.find('bndbox').find('ymax').text
            save_f.writelines('%s %s %s %s %s\n'%(label, xmin, ymin, xmax, ymax))

            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255))
        cv2.imwrite(save_path + '/' + img_id + '.jpg', img)
        save_f.close()


if __name__=="__main__":
    if len(sys.argv) != 6:
        print "Usage: convert image_path list save_path gt save_eg"
        sys.exit()
    convert()
