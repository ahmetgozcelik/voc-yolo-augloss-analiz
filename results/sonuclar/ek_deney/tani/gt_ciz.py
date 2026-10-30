import random
from PIL import Image, ImageDraw
N=["car","van","truck","ped","sit","cyc","tram","misc"]
out=[]
for bol in ("val","train"):
    l=open(f"data/kitti/splits/{bol}.txt").read().split(); random.Random(1).shuffle(l)
    for p in l[:3]:
        im=Image.open(p).convert("RGB"); W,H=im.size; d=ImageDraw.Draw(im)
        lp=p.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
        for s in open(lp).read().splitlines():
            c,x,y,w,h=s.split(); c=int(c); x,y,w,h=map(float,(x,y,w,h))
            d.rectangle([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H],outline=(255,0,0),width=2); d.text(((x-w/2)*W,(y-h/2)*H-10),N[c],fill=(255,255,0))
        d.text((5,5),bol+" "+p.rsplit("/",1)[1],fill=(0,255,0)); out.append(im.resize((828,250)))
M=Image.new("RGB",(828*2,250*3))
for i,im in enumerate(out): M.paste(im,((i//3)*828,(i%3)*250))
M.save(r"<YEREL_YOL>",quality=85)
