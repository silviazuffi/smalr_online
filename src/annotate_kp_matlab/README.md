# Instructions on how to use the annotation tool:

You need Matlab. We have use it with version R2019b.

Edit the file `annotate_images.m` to specify as `base_dir` the directory where you have placed the images.
You can also specify the family of your animal. This information is only used to show the location on the 3D model of the landmarks you are requestd to annotate.

In the annotation tool directory there are several `.mat` files. Each file corresponds to a different set of landmarks.
They mostly differ on the presence of some landmarks on the nose ('wnose' means with nose), on the face ('wface'), or a point at the middle of the tail ('htail').
Choose the one you need. In the fitting code we use the option with face landmarks and with nose and half tail (see the file compute_clips.py).

To annotate:

Run `annotate_images`
The annotations are placed in the `'annotations'` directory in the `'base_dir'` as a `.mat` file that keeps the annotates points so far is created. 



