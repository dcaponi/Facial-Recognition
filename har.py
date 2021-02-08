import json
import glob
import os
import shutil
class Har():
    def __init__(self, target_name):
        print("adding new images from new HAR file")
        hars = glob.glob('*.har')
        if len(hars) > 0:
            print("found {} HAR files".format(len(hars)))
            try:
                os.mkdir(target_name)
            except OSError as e:
                print("target already analyzed, skipping this step")

            for har in hars:
                shutil.move(har, os.path.join(target_name, har))
        else:
            print("No images HARs found")

        self.har_files = glob.glob("{}/*.har".format(target_name))

    def get_media_urls(self, mimetype):
        media_urls = []
        for har_file in self.har_files:
            with open(har_file) as har_json:
                data = json.load(har_json)
                httpTransactions = data['log']['entries']
                for transaction in httpTransactions:
                    if(transaction['request']['url'] and transaction['request']['url'].split('?')[0][-4:] == ('.' + mimetype)):
                        media_urls.append(transaction['request']['url'])

        return media_urls
