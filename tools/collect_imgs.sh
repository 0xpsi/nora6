#!/usr/bin/python

import sys
from google_images_search import GoogleImagesSearch

if len(sys.argv) < 3:
	raise RuntimeError("requires args: <word> <num>")

page = 1
if len(sys.argv) == 4:
	page = int(sys.argv[3])

name = sys.argv[1]
num = int(sys.argv[2])

api = 'G_API_GOES_HERE'
cx = 'G_CUSTOM_CX_GOES_HERE'

gis = GoogleImagesSearch(api, cx)

params = {
	'q': "number " + name,
	'num': num,
	'fileType': 'jpg',
	'safe': 'off',
	'imgSize': 'medium',
	'page': page,
	}

gis.search(search_params=params)
print("Saving", len(gis.results()), "results...")
img = gis.results()
for i in range(len(img)):
	print("(%d\%d) url: %s" % (i+1, len(img), img[i].url))
	img[i].download('images/numbers/' + name)
	#img[i].resize(100,100)

print("Done!")
