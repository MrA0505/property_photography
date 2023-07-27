from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import uuid
from PIL import Image, ImageEnhance

# change .arw to .png

app = Flask(__name__)
@app.route('/process_image', methods=['POST'])
def process_image():
    unique_id = str(uuid.uuid4())
    print(unique_id)
    os.makedirs(f'./tmp/{unique_id}', exist_ok=True)
    os.makedirs('./result', exist_ok=True)
    os.makedirs('./blend_result', exist_ok=True)
    os.makedirs('./gamma_result', exist_ok=True)
    os.makedirs('./final_result', exist_ok=True)
    image_files = []
    for i in range(1, 4):
        image_file = request.files.get(f'image{i}')
        if image_file:
            filename = f'./tmp/{unique_id}/{i}.png'
            image_file.save(filename)
            image_files.append(filename)

    if len(image_files) != 3:
        return jsonify(success=False, message="Three images are required."), 400
    
    img_list = [cv2.imread(fn) for fn in image_files]
    exposure_times = np.array([15.0, 2.5, 0.25], dtype=np.float32)

    merge_debvec = cv2.createMergeDebevec()
    hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy())

    merge_robertson = cv2.createMergeRobertson()
    hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

    tonemap1 = cv2.createTonemapDrago(gamma=2.2)
    res_debvec = tonemap1.process(hdr_debvec.copy())

    tonemap2 = cv2.createTonemapDrago(gamma=1.3)
    res_robertson = tonemap2.process(hdr_robertson.copy())

    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)

    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
    output_filename = f"./result/final_img_{unique_id}.png"

    cv2.imwrite(output_filename, res_mertens_8bit)  
    # ------------------------------------------------------------------

    image1 = cv2.imread(image_files[0]) # Low Exposure
    image2 = cv2.imread(f"./result/final_img_{unique_id}.png")

    if image1.shape != image2.shape:
        height, width = image1.shape[:2]
        image2 = cv2.resize(image2, (width, height))
    
    result = cv2.addWeighted(image1, 0.2, image2, 0.8, 0) # 0.25 and 0.75 is 
    cv2.imwrite(f"./blend_result/blend_img_{unique_id}.png", result)

    # -------------------------------------------------------------------

    source = cv2.imread(f"./blend_result/blend_img_{unique_id}.png")
    corrected = adjust_gamma(source, gamma=1.2)
    adjusted = adjust_saturation(corrected, saturation=1.2)
    cv2.imwrite(f"./gamma_correction/gamma_img_{unique_id}.png", adjusted)

    # -------------------------------------------------------------------
    
    img3 = cv2.imread(image_files[2])
    img4 = cv2.imread(f"./gamma_correction/gamma_img_{unique_id}.png")
    if img3.shape != img4.shape:
        height, width = img3.shape[:2]
        img3 = cv2.resize(img3, (width, height))
    result2 = cv2.addWeighted(img3, 0.6, img4, 0.4, 0)
    cv2.imwrite(f"./blend_result2/blend_img_{unique_id}.png", result2)

    # -------------------------------------------------------------------
    image_gamma = cv2.imread(f"./blend_result2/blend_img_{unique_id}.png")

    gamma = 1.2 # Default gamma = 1.0 
    image_gamma_corrected = np.array(255*(image_gamma / 255) ** gamma, dtype = 'uint8')
    cv2.imwrite(f"./gamma_result/gamma_img_{unique_id}.png", image_gamma_corrected) 
    #-------------------------------------------------------------------------
    gamma_corrected_image = f"./gamma_result/gamma_img_{unique_id}.png"

    output_filename = img_color(gamma_corrected_image, unique_id) 

    for fn in image_files:
        os.remove(fn)
    os.rmdir(f'./tmp/{unique_id}')
    return send_file(output_filename, mimetype='image/png')
    #return 'True'

def img_color(img, unique_id):
    image = Image.open(img)
    enhancer = ImageEnhance.Color(image)
    #image_color = enhancer.enhance(1.0)
    image_color = enhancer.enhance(1.5)  
    image_color.save(f'./final_image/final_output_{unique_id}.png')
    image = f'./final_image/final_output_{unique_id}.png'
    return image
    #return 'Image color Enhancing: True'

def adjust_gamma(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_saturation(image, saturation=1.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype('float32')
    hsv[:,:,1] = hsv[:,:,1] * saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)


if __name__ == '__main__':
    app.run(debug=True)
