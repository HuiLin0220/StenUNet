# all imports required for your model

# device = 'cuda'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
import os, json
import SimpleITK as sitk
from PIL import Image
import numpy as np
from skimage import measure
from shapely.geometry import Polygon
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data as predict
from post_process.remove_small_segments import remove_small_segments

#required functions:
def image_to_array(image_path):
    image = Image.open(image_path)
    array = np.array(image)
    return array
#reading inputs and output folders:

class Prediction_algorithm():

    def __init__(self):

        self.output_file = './coronary-artery-segmentation.json'
        self.weight = './weights/model_final.pth'
        self.output_images_path = './output_images/'
        self.post_output_images_path = './post_output_images/'
        self.model_folder = './model_folder/'
        self.input_images_path = "./saved_images/"
    


    
    def predict_segmentation(self):
        i = 0
        image_list = []
        results_path = self.post_output_images_path
        gt_mask = np.zeros([len(os.listdir(results_path)), 26, 512, 512])
        
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
        
        with open('./ground-truth/ground_truth_segmentation.json') as file:
            gt = json.load(file)              
        
        
        empty_submit = dict()
        empty_submit["images"] = gt["images"]
        empty_submit["categories"] = gt["categories"]
        empty_submit["annotations"] = []
    
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
        
        with open(self.output_file, "w", encoding='ascii') as file:
          json.dump(empty_submit, file)
        
        #return empty_submit
    
    #def predict_segmentation(self):
        # Here you should create your COCO Formatted json file with all annotations, be sure that it satisfies the JSON SCHEMA provided in instructions
        #gt_annotation = open('./ground-truth/ground_truth_segmentation.json')
        #gt_annotation = json.load(gt_annotation)

        #return gt_annotation

    def evaluate(self):

        '''
        # Path to the MHA file
        mha_file = self.input_path
        print(mha_file)
        
        stacked_images = sitk.ReadImage(mha_file)
        
        # Get metadata containing filenames
        # image_filenames = stacked_images.GetMetaData("ImageFilenames").split(';')
        # FYI, metadata was omitted when uploading mha file to Grand Challenge Platform, that's why please use dictionary below

        image_filenames = []
        for i in range(0,30): #or for i in range(0,300) if it is final cases
            image_filenames.append(f"{i}.png")

        mapping_dictionary_preliminary_stenosis = {'slice_4.png': '23.png', 'slice_13.png': '8.png', 'slice_26.png': '4.png', 'slice_7.png': '10.png', 'slice_8.png': '16.png', 'slice_10.png': '9.png', 'slice_9.png': '27.png', 'slice_18.png': '19.png', 'slice_19.png': '26.png', 'slice_6.png': '18.png', 'slice_17.png': '20.png', 'slice_14.png': '6.png', 'slice_21.png': '28.png', 'slice_25.png': '30.png', 'slice_2.png': '14.png', 'slice_5.png': '29.png', 'slice_24.png': '12.png', 'slice_1.png': '2.png', 'slice_23.png': '7.png', 'slice_29.png': '17.png', 'slice_11.png': '15.png', 'slice_20.png': '1.png', 'slice_15.png': '22.png', 'slice_27.png': '3.png', 'slice_3.png': '13.png', 'slice_22.png': '5.png', 'slice_12.png': '11.png', 'slice_0.png': '24.png', 'slice_28.png': '21.png', 'slice_16.png': '25.png'}
        

        # Get pixel data type
        pixel_type = stacked_images.GetPixelIDTypeAsString()

        # Extract and save each image
        for index, filename in enumerate(image_filenames):
            image_array = sitk.GetArrayFromImage(stacked_images)[index]

            # Convert pixel data type to unsigned char if necessary
            if pixel_type != 'siuc8':
                image_array = image_array.astype('uint8')

            image = sitk.GetImageFromArray(image_array)
            # image.SetSpacing(stacked_images.GetSpacing())
            # image.SetOrigin(stacked_images.GetOrigin())
            # image.SetDirection(stacked_images.GetDirection())
            png_name = mapping_dictionary_preliminary_stenosis[f"slice_{filename}"]
            new_name = 'STEN_' + (png_name[:-4]).zfill(3) + '_0000.png'
            
            output_filename = f"./saved_images/{new_name}"
            sitk.WriteImage(image, output_filename)
        '''
        predict(self.input_images_path, self.output_images_path, self.model_folder, [0], 0.5,use_gaussian=True,use_mirroring=True,perform_everything_on_gpu=True,verbose=True,save_probabilities=False,overwrite=False,checkpoint_name=self.weight,num_processes_preprocessing=1,num_processes_segmentation_export=1)
        remove_small_segments(self.output_images_path, self.post_output_images_path, threshold=60)
        # Now, after extracting images you can start your evaluation, make sure format of final json file is correct
        self.predict_segmentation()

        #with open(self.output_file, 'w', encoding='ascii') as f:
                #json.dump(annotations,f)
        
        print("Success with algorithm")
                
        return "Finished"


if __name__ == "__main__":
    
    Prediction_algorithm().evaluate()
