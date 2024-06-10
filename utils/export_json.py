import json
import os
import numpy as np
from datetime import datetime
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon
# Get the current date and time
current_datetime = datetime.today()

# Convert datetime to ISO 8601 format
iso_datetime = current_datetime.isoformat()
def image_to_array(image_path):
    image = Image.open(image_path)
    array = np.array(image)
    return array

def export_json(results_path, output_json_path,empty_json_path):
    num_images = len(os.listdir(results_path))
    with open(empty_json_path) as file:
        gt = json.load(file)

    gt_images = [{
                "id": i+1,
                "width": 512,
                "height": 512,
                "file_name": f"{i+1}.png",
                "license": 0,
                "date_captured": iso_datetime
                }
                for i in range(num_images)]


    info = {'description': 'NU Team 2023',
           'version': 'v1',
           'year': 2023,
           'contributor': 'Hui',
            'date_created': datetime.today().strftime("%Y-%m-%d")}

    licenses = [{'id': i+1,
                 'name': f"{i+1}.png",
                 'url': ""} for i in range(num_images)]

    empty_submit = dict()
    empty_submit["images"] = gt_images
    empty_submit["categories"] = gt["categories"]
    empty_submit["annotations"] = []
    empty_submit["info"] = info
    empty_submit["licenses"] = licenses

    i = 0
    image_list = []
        
    gt_mask = np.zeros([num_images, 26, 512, 512])
        
    for file_name in os.listdir(results_path):
        
            result_path = results_path + file_name
            image_name = str(int(file_name[5:8]))+'.png'
            
            if os.path.exists(result_path):
                prediction_array = image_to_array(result_path)
            else:
                prediction_array = np.zeros((512,512))
            
            image_list.append(image_name)
            gt_mask[i,25] = prediction_array
            i+=1                 
        
    
    image_name_id_dict = {}
    for image in empty_submit["images"]:
            image_name_id_dict[image['file_name']] = image['id']
    
    count_anns = 1
    for img_id, img in enumerate(gt_mask, 0):
            image_name = image_list[img_id] 
            
            
            for cls_id, cls in enumerate(img, 0):
              contours = measure.find_contours(cls)
              for contour in contours:            
                for i in range(len(contour)):
                  row, col = contour[i]
                  contour[i] = (col - 1, row - 1)

                # Simplify polygon
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)
            
                if(poly.is_empty):
                  continue
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
                new_ann = dict()
                new_ann["id"] = count_anns
                new_ann["image_id"] = image_name_id_dict[image_name]
                
                new_ann["category_id"] = cls_id+1
                new_ann["segmentation"] = [segmentation]
                new_ann["area"] = poly.area
                x, y = contour.min(axis=0)
                w, h = contour.max(axis=0) - contour.min(axis=0)
                new_ann["bbox"]  = [int(x), int(y), int(w), int(h)]
                new_ann["iscrowd"] = 0
                new_ann["attributes"] = {
                  "occluded": False
                }
                count_anns += 1
                empty_submit["annotations"].append(new_ann.copy())
        
    print('generating json file')    
    with open(output_json_path, "w", encoding='ascii') as file:
          json.dump(empty_submit, file)
    print(output_json_path)

def test_export_json(output_json_path,empty_json_path):
    num_images = 30
    with open(empty_json_path) as file:
        gt = json.load(file)

    gt_images = [{
                "id": i+1,
                "width": 512,
                "height": 512,
                "file_name": f"{i+1}.png",
                "license": 0,
                "date_captured": iso_datetime
                }
                for i in range(num_images)]


    info = {'description': 'NU Team 2023',
           'version': 'v1',
           'year': 2023,
           'contributor': 'Hui',
            'date_created': datetime.today().strftime("%Y-%m-%d")}

    licenses = [{'id': i+1,
                 'name': f"{i+1}.png",
                 'url': ""} for i in range(num_images)]

    empty_submit = dict()
    empty_submit["images"] = gt_images
    empty_submit["categories"] = gt["categories"]
    empty_submit["annotations"] = []
    empty_submit["info"] = info
    empty_submit["licenses"] = licenses

        
    gt_mask = np.zeros([num_images, 26, 512, 512])        
        
    count_anns = 1
    for img_id, img in enumerate(gt_mask, 0):
           
            for cls_id, cls in enumerate(img, 0):
              contours = measure.find_contours(cls)
              for contour in contours:            
                for i in range(len(contour)):
                  row, col = contour[i]
                  contour[i] = (col - 1, row - 1)

                # Simplify polygon
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)
            
                if(poly.is_empty):
                  continue
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
                new_ann = dict()
                new_ann["id"] = count_anns
                new_ann["image_id"] = img_id+1
                
                new_ann["category_id"] = cls_id+1
                new_ann["segmentation"] = [segmentation]
                new_ann["area"] = poly.area
                x, y = contour.min(axis=0)
                w, h = contour.max(axis=0) - contour.min(axis=0)
                new_ann["bbox"]  = [int(x), int(y), int(w), int(h)]
                new_ann["iscrowd"] = 0
                new_ann["attributes"] = {
                  "occluded": False
                }
                count_anns += 1
                empty_submit["annotations"].append(new_ann.copy())
        
    print('generating json file')    
    with open(output_json_path, "w", encoding='ascii') as file:
          json.dump(empty_submit, file)
    print(output_json_path)
