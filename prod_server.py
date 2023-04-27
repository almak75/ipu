#from fastapi import FastAPI, File, UploadFile,Form, Depends
from fastapi import FastAPI, File, UploadFile

#from typing import Optional
from PIL import Image
from io import BytesIO
#from pydantic import BaseModel#это понадобилось для втрого варианта


import prod #в этом файле хранятся все процедуры

app = FastAPI()

#ПОЛУЧЕНИЕ ТОЛЬКО КАРТИНКИ. ОТВЕТ json

@app.post("/ipu")
async def analyze_image(image: UploadFile = File(...)):
    with Image.open(BytesIO(await image.read())) as img:
        #return {"width": img.width, "height": img.height}
        pok = prod.look_to_file(img)
    return pok
