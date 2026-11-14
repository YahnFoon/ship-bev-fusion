"""Compute pooled and condition-macro COCO AP from exported BEV predictions."""
import argparse
import contextlib
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scripts import evaluate_indoor_20260510_saved_predictions as metrics
KEYS=['coco_mAP','coco_mAP_50','coco_mAP_75']
GEOMETRY_METHODS=['single_223','single_221','single_224','single_222','intersection','vote2','vote3','union']
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def predictions(method,seed,water,gt,m,metric_folder,metric_method,root=None):
 if root is None: raise ValueError('A prediction root is required')
 by_key={(x['condition'],int(x['frame_index'])):x['id'] for x in gt['images']}
 jsonpath=root/'predictions/test/pred_masks_thr0p5_coco.json'
 if method in GEOMETRY_METHODS:jsonpath=root/f'predictions/test/{method}_masks_coco.json'
 if jsonpath.exists():
  data=json.loads(jsonpath.read_text());images={int(x['id']):x for x in data['images']}
  assert len(images)==len(by_key)
  assert {(str(x['condition']),m.frame_index_from_coco_image(x)) for x in images.values()}==set(by_key)
  by_pred={}
  for ann in data['annotations']:
   image=images[int(ann['image_id'])];key=(str(image['condition']),m.frame_index_from_coco_image(image));assert key not in by_pred
   by_pred[key]=ann
  annotations=[]
  for key,image_id in by_key.items():
   if key not in by_pred:continue
   src=by_pred[key];annotations.append(dict(id=len(annotations)+1,image_id=image_id,category_id=1,segmentation=src['segmentation'],area=float(src['area']),bbox=src['bbox'],score=float(src.get('score',1.0))))
  return annotations,dict(format='saved COCO RLE + original confidence; canonical frame ordering',source=str(jsonpath),source_sha256=sha(jsonpath))


 fr=[r for r in read(metric_folder/'frame_level_metrics.csv') if r['method']==metric_method and r['split']=='test']
 scores={(r['condition'],int(r['frame_index'])):float(r['score']) for r in fr};assert set(scores)==set(by_key)
 annotations=[]
 for p in sorted((root/'predictions/test/pred_logits_fp16').glob('*.npz'),key=lambda p:m.condition_key(p.stem)):
  with np.load(p) as f:frames=f['frame_indices'];logits=f['logits']
  assert np.isfinite(logits).all()
  for frame,logit in zip(frames,logits):
   key=(p.stem,int(frame));mask=(logit>=0).astype(np.uint8)
   if not mask.any():continue
   rle=m.encode_binary_mask(mask);annotations.append(dict(id=len(annotations)+1,image_id=by_key[key],category_id=1,segmentation=rle,area=float(mask_utils.area(rle)),bbox=[float(v) for v in mask_utils.toBbox(rle)],score=scores[key]))
  del logits
 return annotations,dict(format='RLE from saved FP16 logits at logit>=0; pinned evaluator frame confidence',source=str(root),score_source=str(metric_folder/'frame_level_metrics.csv'))

def evaluate(gt,pred,m):
 coco=COCO();coco.dataset=gt
 with contextlib.redirect_stdout(io.StringIO()):
  coco.createIndex()
  dt=coco.loadRes(pred) if pred else None
 groups=defaultdict(list)
 for x in gt['images']:groups[x['condition']].append(x['id'])
 def run(ids):
  if not pred:return {k:0. for k in KEYS}
  e=COCOeval(coco,dt,iouType='segm');e.params.imgIds=list(ids)
  with contextlib.redirect_stdout(io.StringIO()):e.evaluate();e.accumulate();e.summarize()
  return {k:float(e.stats[i]) if e.stats[i]>=0 else None for i,k in enumerate(KEYS)}
 pooled=run([x['id'] for x in gt['images']]);rows=[]
 for condition,ids in groups.items():rows.append(dict(condition=condition,frames=len(ids),**run(ids)))


 c=next(iter(groups));ids=set(groups[c]);payload=dict(images=[x for x in gt['images'] if x['id'] in ids],gt_annotations=[a for a in gt['annotations'] if a['image_id'] in ids],pred_annotations=[a for a in pred if a['image_id'] in ids])
 ref=m.coco_ap_metrics_from_payload(payload)
 for k in KEYS:np.testing.assert_allclose(rows[0][k],ref[k],rtol=0,atol=1e-12)
 return pooled,rows

def ground_truth(root, split):
    images=[];annotations=[]
    for path in sorted((root/'targets'/split/'target_masks_uint8').glob('*.npz'), key=lambda p:metrics.condition_key(p.stem)):
        with np.load(path) as data:
            for frame,mask in zip(data['frame_indices'],data['masks']):
                image_id=len(images)+1;h,w=mask.shape
                images.append(dict(id=image_id,file_name=f'{path.stem}/{int(frame):06d}.png',width=w,height=h,condition=path.stem,frame_index=int(frame)))
                if mask.any():
                    rle=metrics.encode_binary_mask(mask.astype(np.uint8))
                    annotations.append(dict(id=len(annotations)+1,image_id=image_id,category_id=1,segmentation=rle,
                        area=float(mask_utils.area(rle)),bbox=mask_utils.toBbox(rle).tolist(),iscrowd=0))
    if not images:raise ValueError('No target masks found')
    return dict(info={},images=images,annotations=annotations,categories=[dict(id=1,name='ship')])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction-root',type=Path,required=True)
    p.add_argument('--method',default='ours')
    p.add_argument('--metrics-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();gt=ground_truth(a.prediction_root,'test')
    pred,_=predictions(a.method,None,None,gt,metrics,a.metrics_dir,a.method,a.prediction_root)
    pooled,rows=evaluate(gt,pred,metrics)
    summary={}
    for key in KEYS:
        values=np.asarray([r[key] for r in rows if r[key] is not None],dtype=float)
        summary[key]=dict(condition_mean=float(values.mean()) if len(values) else None,
            condition_sd=float(values.std(ddof=1)) if len(values)>1 else None,
            valid_conditions=len(values),total_conditions=len(rows),pooled=pooled[key])
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.output/'condition_ap.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['condition','frames',*KEYS]);w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
