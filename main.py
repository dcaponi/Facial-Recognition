from multiprocessing import Pool
from har import Har
from face_finder import FaceFinder
import requests
import os
import sys
import glob

def request_img_data(url):
    try:
        response_stream = requests.get(url, stream=True)
        response_stream.raw.decode_content = True
        return response_stream.content
    except socket.timeout:
        print("Couldn't connect to", url)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        action = sys.argv[1]
        target_name = sys.argv[2]
    else:
        print("An action [add_images (init), train] and a target_name are both requierd")

    if action == "add_images" or action == "init":
        img_urls = Har(target_name).get_media_urls('jpg')
        pool = Pool(processes=6)
        images_data = pool.map(request_img_data, img_urls)
        face_finder = FaceFinder(target_name)

        for i in range(len(images_data)):
            face_finder.save_face_from_data("{}.jpg".format(i), images_data[i])

        print("Target images added for {}. Please double check that all these faces are in fact your target".format(target_name))

    elif action == "train":
        print("training on {}".format(target_name))
        face_finder = FaceFinder(target_name)
        face_finder.learn_target()

    elif action == "scan_for":
        if len(sys.argv) < 3:
            print("You must provide a url")
            quit()
        url = sys.argv[3]
        print("Scanning picture for {}".format(target_name))
        data = request_img_data(url)
        face_finder = FaceFinder(target_name)
        face_finder.scan_pic_for(data)
        
    else:
        print("invaid action {} given".format(action))
