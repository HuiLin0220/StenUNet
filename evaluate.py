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
from utils.utils import mkdir

#required functions:
def image_to_array(image_path):
    image = Image.open(image_path)
    array = np.array(image)
    return array
#reading inputs and output folders:

class Prediction_algorithm():

    def __init__(self):
        folderpath_read = r'/input/images/coronary-angiography-x-ray-stack/'
        filename = os.listdir(folderpath_read)[0]  # pick the first (and only) file in folder
        self.input_path = os.path.join(folderpath_read, filename)

        folderpath_write = r'/output/'
        output_filename  = 'coronary-artery-segmentation.json'
        self.output_file = os.path.join(folderpath_write, output_filename)
        self.weight = '/opt/app/weights/model_final.pth'
        self.output_images_path = '/opt/app/output_images/'
        mkdir(self.output_images_path)
        self.post_output_images_path = '/opt/app/post_output_images/'
        mkdir(self.post_output_images_path)
        self.model_folder = '/opt/app/model_folder/'
        self.input_images_path = "/opt/app/saved_images/"
        mkdir(self.input_images_path)
          
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
        
        with open('/opt/app/ground-truth/ground_truth_segmentation.json') as file:
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
        #gt_annotation = open('/opt/app/ground-truth/ground_truth_segmentation.json')
        #gt_annotation = json.load(gt_annotation)

        #return gt_annotation

    def evaluate(self):


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

        mapping_dictionary_preliminary_segmentation = {'slice_4.png': '23.png', 'slice_13.png': '8.png', 'slice_26.png': '4.png', 'slice_7.png': '10.png', 'slice_8.png': '16.png', 'slice_10.png': '9.png', 'slice_9.png': '27.png', 'slice_18.png': '19.png', 'slice_19.png': '26.png', 'slice_6.png': '18.png', 'slice_17.png': '20.png', 'slice_14.png': '6.png', 'slice_21.png': '28.png', 'slice_25.png': '30.png', 'slice_2.png': '14.png', 'slice_5.png': '29.png', 'slice_24.png': '12.png', 'slice_1.png': '2.png', 'slice_23.png': '7.png', 'slice_29.png': '17.png', 'slice_11.png': '15.png', 'slice_20.png': '1.png', 'slice_15.png': '22.png', 'slice_27.png': '3.png', 'slice_3.png': '13.png', 'slice_22.png': '5.png', 'slice_12.png': '11.png', 'slice_0.png': '24.png', 'slice_28.png': '21.png', 'slice_16.png': '25.png'}
        mapping_dictionary_preliminary_stenosis = {'slice_4.png': '23.png', 'slice_13.png': '8.png', 'slice_26.png': '4.png', 'slice_7.png': '10.png', 'slice_8.png': '16.png', 'slice_10.png': '9.png', 'slice_9.png': '27.png', 'slice_18.png': '19.png', 'slice_19.png': '26.png', 'slice_6.png': '18.png', 'slice_17.png': '20.png', 'slice_14.png': '6.png', 'slice_21.png': '28.png', 'slice_25.png': '30.png', 'slice_2.png': '14.png', 'slice_5.png': '29.png', 'slice_24.png': '12.png', 'slice_1.png': '2.png', 'slice_23.png': '7.png', 'slice_29.png': '17.png', 'slice_11.png': '15.png', 'slice_20.png': '1.png', 'slice_15.png': '22.png', 'slice_27.png': '3.png', 'slice_3.png': '13.png', 'slice_22.png': '5.png', 'slice_12.png': '11.png', 'slice_0.png': '24.png', 'slice_28.png': '21.png', 'slice_16.png': '25.png'}
        mapping_dictionary_final_segmentation = {'slice_210.png': '264.png', 'slice_141.png': '30.png', 'slice_221.png': '1.png', 'slice_140.png': '106.png', 'slice_274.png': '152.png', 'slice_103.png': '165.png', 'slice_241.png': '109.png', 'slice_37.png': '189.png', 'slice_190.png': '254.png', 'slice_236.png': '20.png', 'slice_256.png': '259.png', 'slice_36.png': '73.png', 'slice_294.png': '98.png', 'slice_73.png': '92.png', 'slice_53.png': '283.png', 'slice_181.png': '211.png', 'slice_162.png': '44.png', 'slice_273.png': '29.png', 'slice_60.png': '280.png', 'slice_220.png': '278.png', 'slice_149.png': '31.png', 'slice_281.png': '167.png', 'slice_105.png': '145.png', 'slice_62.png': '252.png', 'slice_4.png': '80.png', 'slice_111.png': '212.png', 'slice_115.png': '41.png', 'slice_295.png': '205.png', 'slice_128.png': '168.png', 'slice_131.png': '194.png', 'slice_226.png': '127.png', 'slice_219.png': '290.png', 'slice_257.png': '76.png', 'slice_154.png': '10.png', 'slice_133.png': '197.png', 'slice_94.png': '141.png', 'slice_216.png': '74.png', 'slice_47.png': '296.png', 'slice_161.png': '4.png', 'slice_222.png': '43.png', 'slice_13.png': '243.png', 'slice_102.png': '225.png', 'slice_86.png': '232.png', 'slice_132.png': '134.png', 'slice_165.png': '285.png', 'slice_242.png': '123.png', 'slice_174.png': '200.png', 'slice_42.png': '251.png', 'slice_144.png': '36.png', 'slice_121.png': '219.png', 'slice_264.png': '226.png', 'slice_39.png': '14.png', 'slice_231.png': '50.png', 'slice_26.png': '183.png', 'slice_136.png': '12.png', 'slice_247.png': '99.png', 'slice_175.png': '231.png', 'slice_7.png': '121.png', 'slice_8.png': '68.png', 'slice_244.png': '52.png', 'slice_206.png': '11.png', 'slice_83.png': '258.png', 'slice_10.png': '292.png', 'slice_289.png': '60.png', 'slice_159.png': '116.png', 'slice_268.png': '162.png', 'slice_9.png': '57.png', 'slice_280.png': '100.png', 'slice_18.png': '7.png', 'slice_19.png': '32.png', 'slice_286.png': '181.png', 'slice_33.png': '237.png', 'slice_35.png': '34.png', 'slice_110.png': '77.png', 'slice_285.png': '154.png', 'slice_177.png': '91.png', 'slice_158.png': '71.png', 'slice_151.png': '132.png', 'slice_106.png': '187.png', 'slice_123.png': '15.png', 'slice_272.png': '267.png', 'slice_157.png': '163.png', 'slice_200.png': '239.png', 'slice_55.png': '216.png', 'slice_213.png': '178.png', 'slice_269.png': '222.png', 'slice_233.png': '240.png', 'slice_152.png': '282.png', 'slice_80.png': '265.png', 'slice_31.png': '244.png', 'slice_100.png': '174.png', 'slice_179.png': '107.png', 'slice_234.png': '72.png', 'slice_188.png': '276.png', 'slice_196.png': '279.png', 'slice_160.png': '117.png', 'slice_246.png': '125.png', 'slice_6.png': '95.png', 'slice_287.png': '84.png', 'slice_68.png': '193.png', 'slice_203.png': '281.png', 'slice_113.png': '33.png', 'slice_130.png': '105.png', 'slice_224.png': '185.png', 'slice_199.png': '108.png', 'slice_259.png': '270.png', 'slice_166.png': '227.png', 'slice_283.png': '115.png', 'slice_207.png': '39.png', 'slice_218.png': '87.png', 'slice_284.png': '202.png', 'slice_119.png': '22.png', 'slice_227.png': '9.png', 'slice_138.png': '175.png', 'slice_296.png': '147.png', 'slice_135.png': '299.png', 'slice_167.png': '104.png', 'slice_95.png': '18.png', 'slice_143.png': '221.png', 'slice_137.png': '176.png', 'slice_63.png': '86.png', 'slice_75.png': '27.png', 'slice_239.png': '245.png', 'slice_164.png': '140.png', 'slice_67.png': '124.png', 'slice_30.png': '249.png', 'slice_17.png': '268.png', 'slice_183.png': '93.png', 'slice_126.png': '217.png', 'slice_79.png': '81.png', 'slice_66.png': '58.png', 'slice_14.png': '261.png', 'slice_82.png': '54.png', 'slice_117.png': '182.png', 'slice_278.png': '118.png', 'slice_266.png': '56.png', 'slice_176.png': '110.png', 'slice_139.png': '218.png', 'slice_291.png': '59.png', 'slice_107.png': '129.png', 'slice_202.png': '79.png', 'slice_109.png': '157.png', 'slice_49.png': '274.png', 'slice_21.png': '28.png', 'slice_44.png': '164.png', 'slice_89.png': '143.png', 'slice_124.png': '204.png', 'slice_282.png': '160.png', 'slice_45.png': '210.png', 'slice_122.png': '229.png', 'slice_120.png': '155.png', 'slice_212.png': '148.png', 'slice_208.png': '286.png', 'slice_163.png': '297.png', 'slice_229.png': '46.png', 'slice_38.png': '146.png', 'slice_261.png': '262.png', 'slice_248.png': '220.png', 'slice_25.png': '13.png', 'slice_194.png': '40.png', 'slice_155.png': '184.png', 'slice_34.png': '166.png', 'slice_88.png': '3.png', 'slice_90.png': '65.png', 'slice_146.png': '298.png', 'slice_209.png': '102.png', 'slice_182.png': '159.png', 'slice_255.png': '23.png', 'slice_184.png': '188.png', 'slice_2.png': '153.png', 'slice_5.png': '122.png', 'slice_293.png': '180.png', 'slice_24.png': '201.png', 'slice_1.png': '114.png', 'slice_125.png': '196.png', 'slice_23.png': '8.png', 'slice_54.png': '234.png', 'slice_58.png': '161.png', 'slice_76.png': '82.png', 'slice_195.png': '242.png', 'slice_240.png': '90.png', 'slice_198.png': '45.png', 'slice_29.png': '69.png', 'slice_74.png': '88.png', 'slice_52.png': '150.png', 'slice_243.png': '256.png', 'slice_96.png': '96.png', 'slice_186.png': '233.png', 'slice_172.png': '273.png', 'slice_11.png': '139.png', 'slice_46.png': '215.png', 'slice_70.png': '133.png', 'slice_145.png': '144.png', 'slice_108.png': '64.png', 'slice_251.png': '16.png', 'slice_197.png': '131.png', 'slice_169.png': '62.png', 'slice_20.png': '49.png', 'slice_252.png': '255.png', 'slice_193.png': '78.png', 'slice_99.png': '177.png', 'slice_84.png': '300.png', 'slice_92.png': '126.png', 'slice_267.png': '260.png', 'slice_279.png': '209.png', 'slice_97.png': '38.png', 'slice_59.png': '94.png', 'slice_232.png': '295.png', 'slice_148.png': '138.png', 'slice_265.png': '213.png', 'slice_237.png': '156.png', 'slice_15.png': '224.png', 'slice_271.png': '214.png', 'slice_116.png': '6.png', 'slice_191.png': '170.png', 'slice_51.png': '235.png', 'slice_230.png': '21.png', 'slice_214.png': '275.png', 'slice_27.png': '190.png', 'slice_235.png': '179.png', 'slice_65.png': '128.png', 'slice_3.png': '25.png', 'slice_48.png': '151.png', 'slice_101.png': '253.png', 'slice_170.png': '119.png', 'slice_245.png': '37.png', 'slice_114.png': '5.png', 'slice_254.png': '75.png', 'slice_250.png': '223.png', 'slice_277.png': '112.png', 'slice_298.png': '171.png', 'slice_98.png': '63.png', 'slice_290.png': '17.png', 'slice_40.png': '101.png', 'slice_178.png': '236.png', 'slice_71.png': '85.png', 'slice_22.png': '192.png', 'slice_43.png': '120.png', 'slice_173.png': '241.png', 'slice_215.png': '228.png', 'slice_263.png': '246.png', 'slice_56.png': '257.png','slice_260.png':'288.png', 'slice_69.png': '195.png', 'slice_288.png': '238.png', 'slice_189.png': '198.png', 'slice_129.png': '42.png', 'slice_77.png': '191.png', 'slice_297.png': '103.png','slice_93.png':'48.png', 'slice_201.png': '55.png', 'slice_91.png': '248.png', 'slice_87.png': '137.png', 'slice_104.png': '2.png', 'slice_112.png': '135.png', 'slice_258.png': '208.png', 'slice_187.png': '266.png', 'slice_78.png': '247.png', 'slice_299.png': '230.png', 'slice_185.png': '61.png', 'slice_270.png': '206.png', 'slice_156.png': '203.png', 'slice_12.png': '130.png', 'slice_276.png': '111.png', 'slice_57.png': '113.png', 'slice_180.png': '89.png', 'slice_134.png': '136.png', 'slice_118.png': '291.png', 'slice_217.png': '173.png', 'slice_0.png': '199.png', 'slice_28.png': '271.png', 'slice_238.png': '70.png', 'slice_81.png': '169.png', 'slice_41.png': '149.png', 'slice_253.png': '250.png', 'slice_127.png': '24.png', 'slice_153.png': '289.png', 'slice_50.png': '53.png', 'slice_249.png': '26.png', 'slice_85.png': '35.png', 'slice_225.png': '51.png', 'slice_192.png': '83.png', 'slice_262.png': '67.png', 'slice_32.png': '287.png', 'slice_211.png': '142.png', 'slice_64.png': '207.png', 'slice_171.png': '294.png', 'slice_228.png': '19.png', 'slice_147.png': '284.png', 'slice_205.png': '293.png', 'slice_275.png': '272.png', 'slice_223.png': '97.png', 'slice_61.png': '186.png', 'slice_72.png': '277.png', 'slice_292.png': '269.png', 'slice_168.png': '47.png', 'slice_16.png': '66.png', 'slice_204.png': '158.png', 'slice_142.png': '263.png', 'slice_150.png': '172.png'}
        mapping_dictionary_final_stenosis = {'slice_210.png': '264.png', 'slice_141.png': '30.png', 'slice_221.png': '1.png', 'slice_140.png': '106.png', 'slice_105.png':'145.png','slice_274.png': '152.png', 'slice_103.png': '165.png', 'slice_241.png': '109.png', 'slice_37.png': '189.png', 'slice_190.png': '254.png', 'slice_236.png': '20.png', 'slice_256.png': '259.png', 'slice_36.png': '73.png', 'slice_294.png': '98.png', 'slice_73.png': '92.png', 'slice_53.png': '283.png', 'slice_181.png': '211.png', 'slice_162.png': '44.png', 'slice_273.png': '29.png', 'slice_60.png': '280.png', 'slice_220.png': '278.png', 'slice_149.png': '31.png', 'slice_281.png': '167.png', 'slice_192.png': '83.png', 'slice_62.png': '252.png', 'slice_4.png': '80.png', 'slice_46.png': '215.png', 'slice_115.png': '41.png', 'slice_295.png': '205.png', 'slice_128.png': '168.png', 'slice_111.png':'212.png','slice_131.png': '194.png', 'slice_226.png': '127.png', 'slice_219.png': '290.png', 'slice_257.png': '76.png', 'slice_154.png': '10.png', 'slice_133.png': '197.png', 'slice_94.png': '141.png', 'slice_216.png': '74.png', 'slice_47.png': '296.png', 'slice_161.png': '4.png', 'slice_222.png': '43.png', 'slice_13.png': '243.png', 'slice_85.png':'35.png','slice_102.png': '225.png', 'slice_86.png': '232.png', 'slice_132.png': '134.png', 'slice_165.png': '285.png', 'slice_242.png': '123.png', 'slice_174.png': '200.png', 'slice_42.png': '251.png', 'slice_144.png': '36.png', 'slice_121.png': '219.png', 'slice_264.png': '226.png', 'slice_39.png': '14.png', 'slice_231.png': '50.png', 'slice_26.png': '183.png', 'slice_136.png': '12.png', 'slice_247.png': '99.png', 'slice_175.png': '231.png', 'slice_7.png': '121.png', 'slice_8.png': '68.png', 'slice_244.png': '52.png', 'slice_206.png': '11.png','slice_153.png':'289.png' ,'slice_83.png': '258.png', 'slice_10.png': '292.png', 'slice_289.png': '60.png', 'slice_159.png': '116.png', 'slice_268.png': '162.png', 'slice_9.png': '57.png', 'slice_280.png': '100.png', 'slice_18.png': '7.png', 'slice_255.png':'23.png','slice_19.png': '32.png', 'slice_286.png': '181.png', 'slice_33.png': '237.png', 'slice_35.png': '34.png', 'slice_110.png': '77.png', 'slice_285.png': '154.png', 'slice_177.png': '91.png', 'slice_158.png': '71.png', 'slice_151.png': '132.png', 'slice_106.png': '187.png', 'slice_123.png': '15.png', 'slice_272.png': '267.png', 'slice_157.png': '163.png', 'slice_200.png': '239.png', 'slice_55.png': '216.png', 'slice_213.png': '178.png', 'slice_269.png': '222.png', 'slice_233.png': '240.png', 'slice_152.png': '282.png', 'slice_80.png': '265.png', 'slice_31.png': '244.png', 'slice_100.png': '174.png', 'slice_179.png': '107.png', 'slice_234.png': '72.png', 'slice_188.png': '276.png', 'slice_196.png': '279.png', 'slice_160.png': '117.png', 'slice_246.png': '125.png', 'slice_6.png': '95.png', 'slice_287.png': '84.png', 'slice_68.png': '193.png', 'slice_203.png': '281.png', 'slice_113.png': '33.png', 'slice_130.png': '105.png', 'slice_224.png': '185.png', 'slice_199.png': '108.png', 'slice_259.png': '270.png', 'slice_166.png': '227.png', 'slice_283.png': '115.png', 'slice_207.png': '39.png', 'slice_218.png': '87.png', 'slice_284.png': '202.png', 'slice_119.png': '22.png', 'slice_227.png': '9.png', 'slice_138.png': '175.png', 'slice_296.png': '147.png', 'slice_135.png': '299.png', 'slice_167.png': '104.png', 'slice_95.png': '18.png', 'slice_143.png': '221.png', 'slice_137.png': '176.png', 'slice_63.png': '86.png', 'slice_75.png': '27.png', 'slice_239.png': '245.png', 'slice_164.png': '140.png', 'slice_67.png': '124.png', 'slice_30.png': '249.png', 'slice_17.png': '268.png', 'slice_183.png': '93.png', 'slice_126.png': '217.png', 'slice_79.png': '81.png', 'slice_66.png': '58.png', 'slice_14.png': '261.png', 'slice_82.png': '54.png', 'slice_117.png': '182.png', 'slice_260.png': '288.png', 'slice_93.png': '48.png', 'slice_278.png': '118.png', 'slice_266.png': '56.png', 'slice_176.png': '110.png', 'slice_139.png': '218.png', 'slice_291.png': '59.png', 'slice_107.png': '129.png', 'slice_202.png': '79.png', 'slice_109.png': '157.png', 'slice_49.png': '274.png', 'slice_21.png': '28.png', 'slice_44.png': '164.png', 'slice_89.png': '143.png', 'slice_124.png': '204.png', 'slice_282.png': '160.png', 'slice_45.png': '210.png', 'slice_122.png': '229.png', 'slice_120.png': '155.png', 'slice_212.png': '148.png', 'slice_208.png': '286.png', 'slice_163.png': '297.png', 'slice_229.png': '46.png', 'slice_38.png': '146.png', 'slice_261.png': '262.png', 'slice_248.png': '220.png', 'slice_25.png': '13.png', 'slice_194.png': '40.png', 'slice_155.png': '184.png', 'slice_34.png': '166.png', 'slice_88.png': '3.png', 'slice_90.png': '65.png', 'slice_146.png': '298.png', 'slice_209.png': '102.png', 'slice_182.png': '159.png', 'slice_81.png': '169.png', 'slice_184.png': '188.png', 'slice_2.png': '153.png', 'slice_5.png': '122.png', 'slice_293.png': '180.png', 'slice_24.png': '201.png', 'slice_1.png': '114.png', 'slice_125.png': '196.png', 'slice_23.png': '8.png', 'slice_54.png': '234.png', 'slice_58.png': '161.png', 'slice_76.png': '82.png', 'slice_195.png': '242.png', 'slice_240.png': '90.png', 'slice_198.png': '45.png', 'slice_29.png': '69.png', 'slice_74.png': '88.png', 'slice_52.png': '150.png', 'slice_243.png': '256.png', 'slice_96.png': '96.png', 'slice_186.png': '233.png', 'slice_172.png': '273.png', 'slice_11.png': '139.png', 'slice_70.png': '133.png', 'slice_145.png': '144.png', 'slice_108.png': '64.png', 'slice_251.png': '16.png', 'slice_197.png': '131.png', 'slice_169.png': '62.png', 'slice_20.png': '49.png', 'slice_252.png': '255.png', 'slice_193.png': '78.png', 'slice_99.png': '177.png', 'slice_84.png': '300.png', 'slice_92.png': '126.png', 'slice_267.png': '260.png', 'slice_279.png': '209.png', 'slice_97.png': '38.png', 'slice_59.png': '94.png', 'slice_232.png': '295.png', 'slice_148.png': '138.png', 'slice_265.png': '213.png', 'slice_237.png': '156.png', 'slice_15.png': '224.png', 'slice_271.png': '214.png', 'slice_116.png': '6.png', 'slice_191.png': '170.png', 'slice_51.png': '235.png', 'slice_230.png': '21.png', 'slice_214.png': '275.png', 'slice_27.png': '190.png', 'slice_235.png': '179.png', 'slice_65.png': '128.png', 'slice_3.png': '25.png', 'slice_48.png': '151.png', 'slice_101.png': '253.png', 'slice_170.png': '119.png', 'slice_245.png': '37.png', 'slice_114.png': '5.png', 'slice_254.png': '75.png', 'slice_250.png': '223.png', 'slice_277.png': '112.png', 'slice_298.png': '171.png', 'slice_98.png': '63.png', 'slice_290.png': '17.png', 'slice_40.png': '101.png', 'slice_178.png': '236.png', 'slice_71.png': '85.png', 'slice_22.png': '192.png', 'slice_43.png': '120.png', 'slice_173.png': '241.png', 'slice_215.png': '228.png', 'slice_263.png': '246.png', 'slice_56.png': '257.png', 'slice_69.png': '195.png', 'slice_288.png': '238.png', 'slice_189.png': '198.png', 'slice_129.png': '42.png', 'slice_77.png': '191.png', 'slice_297.png': '103.png', 'slice_201.png': '55.png', 'slice_91.png': '248.png', 'slice_87.png': '137.png', 'slice_104.png': '2.png', 'slice_112.png': '135.png', 'slice_258.png': '208.png', 'slice_187.png': '266.png', 'slice_78.png': '247.png', 'slice_299.png': '230.png', 'slice_185.png': '61.png', 'slice_270.png': '206.png', 'slice_156.png': '203.png', 'slice_12.png': '130.png', 'slice_276.png': '111.png', 'slice_57.png': '113.png', 'slice_180.png': '89.png', 'slice_134.png': '136.png', 'slice_118.png': '291.png', 'slice_217.png': '173.png', 'slice_0.png': '199.png', 'slice_28.png': '271.png', 'slice_238.png': '70.png', 'slice_41.png': '149.png', 'slice_253.png': '250.png', 'slice_127.png': '24.png', 'slice_64.png': '207.png', 'slice_50.png': '53.png', 'slice_249.png': '26.png', 'slice_225.png': '51.png', 'slice_262.png': '67.png', 'slice_32.png': '287.png', 'slice_211.png': '142.png', 'slice_171.png': '294.png', 'slice_228.png': '19.png', 'slice_147.png': '284.png', 'slice_205.png': '293.png', 'slice_275.png': '272.png', 'slice_223.png': '97.png', 'slice_61.png': '186.png', 'slice_72.png': '277.png', 'slice_292.png': '269.png', 'slice_168.png': '47.png', 'slice_16.png': '66.png', 'slice_204.png': '158.png', 'slice_142.png': '263.png', 'slice_150.png': '172.png'}

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
            
            output_filename = f"/opt/app/saved_images/{new_name}"
            sitk.WriteImage(image, output_filename)
        
        predict(self.input_images_path, self.output_images_path, self.model_folder, [0], 0.5,use_gaussian=True,use_mirroring=True,perform_everything_on_gpu=True,verbose=True,save_probabilities=False,overwrite=False,checkpoint_name=self.weight,num_processes_preprocessing=1,num_processes_segmentation_export=1)
        
        remove_small_segments(self.output_images_path, self.post_output_images_path, threshold = 60)
        
        # Now, after extracting images you can start your evaluation, make sure format of final json file is correct
        self.predict_segmentation()

        #with open(self.output_file, 'w', encoding='ascii') as f:
                #json.dump(annotations,f)
        
        print("Success with algorithm")
                
        return "Finished"


if __name__ == "__main__":
    
    Prediction_algorithm().evaluate()
