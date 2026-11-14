"""Validate synchronized frame membership and COCO annotation references."""
import argparse
import json
from pathlib import Path


def validate(root):
    rows=[]; masks=0;annotations=0
    for water in [9,11,13]:
        m=json.loads((root/f'indoor/splits/water{water}.json').read_text())
        used=set()
        for split,part in m['splits'].items():
            conditions=part['conditions']
            assert not (used & set(conditions)), 'Condition leakage between splits'
            used.update(conditions)
            assert set(conditions)==set(part['frames_by_condition'])
            count=0
            for c in conditions:
                frames=part['frames_by_condition'][c]
                assert len(frames)==len(set(frames)) and frames==sorted(frames)
                count+=len(frames)
                for cam in ['221','222','223','224','GT']:
                    video=root/'indoor'/m['video_root']/cam/f'camera_{cam}_{c}.mp4'
                    assert video.is_file() and video.stat().st_size>0
                    path=root/'indoor'/m['mask_root']/f'view_{cam}/condition_{c}/sam2/camera_{cam}_{c}_coco_masks.json'
                    data=json.loads(path.read_text())
                    images={int(x['id']):x for x in data['images']}
                    assert len(images)==len(data['images'])
                    assert set(frames)<=set(images), (water,c,cam,'missing mask frames')
                    ids=set()
                    for ann in data['annotations']:
                        assert ann['id'] not in ids
                        ids.add(ann['id'])
                        image=images[int(ann['image_id'])]
                        seg=ann['segmentation']
                        assert seg['size']==[image['height'],image['width']]
                        assert isinstance(seg['counts'],(str,list))
                    masks+=1;annotations+=len(data['annotations'])
            rows.append(dict(water_level_mm=m['bev']['plane_z_mm'],split=split,conditions=len(conditions),synchronized_frames=count))
    return dict(status='passed', splits=rows, indoor_mask_files=masks, indoor_annotations=annotations)



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('data'));p.add_argument('--output',type=Path)
    a=p.parse_args();r=validate(a.root)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
