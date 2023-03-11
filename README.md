# KaliYuga's DreamBooth With Dataset Captioning


This is KaliYuga's fork of Shivam Shrirao's [DreamBooth implementation](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth). It adds a number of new features to make dataset labeling and organization faster and more powerful, and training more accurate (hopefully!).

**This fork adds the following:** 

*   a slightly modified version of the ***BLIP dataset
autocaptioning functionality*** from [victorchall's EveryDream comapnion tools repo](https://github.com/victorchall/EveryDream).

Once you've autocaptioned your datasets, you can use this same notebook to train Stable Diffusion models on your new text/image pairs (with or without instance and class prompts) using 

*   ***KaliYuga's Dataset Organization Omnitool***, which I wrote with copious help from ChatGPT. This tool lets you extract .txt files from your image filenames so you can train on unique text/image pairs instead of using a single broad instance prompt for a whole group of images. You can still use class and instance prompts alongside the text/image pairs if you want, and this can be a good way to hyper-organize your training data. More detail is given in the Omnitool section.

------
You can support victorchall's awesome EveryDream on 
[Patreon](https://www.patreon.com/everydream) or at [Kofi](https://ko-fi.com/everydream)!

[Follow Shivam](https://twitter.com/ShivamShrirao) on Twitter

[Follow me](https://twitter.com/KaliYuga_ai) (KaliYuga) on Twitter!

--------

**Changelog**
**March 10, '23**
Added support for .gif and .jpeg autocaption extraction in the "Run" section under the Omnitool.

**Feb 26, '23** 
* Updated the prompt input and text file extraction section to skip generation of new text files for images if an accurate text file already exists.
    * Added a text log of of text file generation/skipping and an accurate progress bar
* Added a note that class images are needed (for now) in both modes. If you don't add class images/reg images, it tries to read the train_dreambooth.py file as an image, and the whole thing breaks
* Fixed an issue in which an extra /content/drive/MyDrive was prepended to OUTPUT_DIR.

-------

## ** About KaliYuga's Dataset Organization Omnitool**
# Methods of Use

**Method 1: Use Image Filenames As Instance Prompts**
***What Method 1 Does***
When you specify the path to your image dataset, running the cell creates text files of each image caption (filename). These are then used instead of instance prompts. 

**To use this**, simply do not input instance/class prompts or class path below. **You will still need to input an instance_directory path and a class directory**, as this is the path to all your dataset images and regularization images.
<br></br>

**Method 2: Use Image Filenames alsongside Instance Prompts and Class Prompts**

***What Method 2 Does***
Like method one, this method extracts the filenames of all the images in a specified directory and saves them to a text file which can be used as input for machine learning models. Unlike the above section, though, you can use **instance prompts** and **class prompts** alongside basic extracted image captions.
<br></br>

***What Are Instance and Class Prompts?***

Instance and class prompts are additional text descriptions of the image content that can help improve the quality of the model's output.

Instance prompts describe specific objects or features within an image, such as **"a red car"** or **"a smiling person."** Class prompts describe more general concepts or categories, such as **"a car"** or **"a person."** By including these prompts in the training data, the model can learn to associate specific features with broader categories, resulting in more accurate and nuanced results. 

Please note that, if you're using class prompts, you do not have to provide class images yourself. The class prompt you provide will be used to generate these automatically from stable diffusion and save them to your drive.
<br></br>
**IMPORTANT NOTE:**

If you have the same word in both your instance and class prompt, it can lead to overfitting on that specific word. When this happens, the model may focus too heavily on that word and generate images that only match that word, rather than the overall concept. To avoid this, it's recommended to choose unique and distinct prompts for both the instance and class.
<br></br>

**Pros and Cons of These Methods of Dataset Labeling**

***Pros--***much more accurate text/image pairs than a general instance prompt (assuming good image captioning)

***Cons--***hand-captioning datasets is slow going, and tools like BLIP are not always accurate.
