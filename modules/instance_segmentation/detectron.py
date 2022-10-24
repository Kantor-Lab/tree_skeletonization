from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
import cv2

class detectron_predictor():
    def __init__(self,
                 config_file='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
                 weights_path='model_final.pth',
                 num_classes=1, 
                 score_threshold=0.5):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.WEIGHTS = weights_path 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold 
        self.predictor = DefaultPredictor(cfg)

    def predict(self, img, visualize=False):        
        #img = cv2.imread(os.path.join(test_dir, d))
        img_in = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        outputs = self.predictor(img_in)
        masks = outputs['instances'].get('pred_masks')
        masks = masks.to('cpu').numpy()
        scores = outputs['instances'].get('scores')
        scores = scores.to('cpu').numpy()
        for i in range(2): # Repeat just in case
            masks, scores = self.filter_overlapping_masks(masks, scores)
        if visualize:
            v = Visualizer(img[:, :, ::-1], scale=0.8)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            img_vis = v.get_image()[:, :, ::-1]
            from PIL import Image
            im = Image.fromarray(img_vis)
            im.show()
            #plt.imshow(img_vis)
            #plt.axis('off')
            #plt.show()
        return masks, scores
    
    @staticmethod
    def filter_overlapping_masks(masks, scores):
        new_masks = []
        new_scores = []
        n_masks = len(masks)
        joined_masks = []
        for i in range(n_masks):
            for j in range(i+1, n_masks):
                if j in joined_masks:
                    break
                num_mask_i = np.sum(masks[i])
                num_mask_j = np.sum(masks[j])
                num_mask_ij = np.sum(np.logical_and(masks[i], masks[j]))
                if num_mask_ij/num_mask_i>0.5 or num_mask_ij/num_mask_j>0.5:
                    new_mask = np.logical_or(masks[i], masks[j])
                    new_masks.append(new_mask)
                    joined_masks.append(i)
                    joined_masks.append(j)
                    new_scores.append((scores[i]+scores[j])/2)
                    continue
            if i in joined_masks:
                continue
            new_masks.append(masks[i])
            joined_masks.append(i)
            new_scores.append(scores[i])
        new_masks = np.array(new_masks)
        new_scores = np.array(new_scores)
        return new_masks, new_scores


if __name__=='__main__':
    img_path = '/home/johnkim/catkin_ws/src/pointcloud_segmentation/synthetic_tree.png'

    DETECTRON = detectron_predictor(
        config_file='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        weights_path='models/detectron_branch_segmentation.pth',
        num_classes=1, 
        score_threshold=0.3
    )
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    branch_masks, scores = DETECTRON.predict(img, visualize=True)



    

