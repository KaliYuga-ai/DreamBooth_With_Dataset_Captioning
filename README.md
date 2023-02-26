# DreamBooth With Dataset Captioning

# KaliYuga's DreamBooth With Dataset Captioning

<div>
<img src="https://images.squarespace-cdn.com/content/v1/6213c340453c3f502425776e/a432c21c-bb12-4f38-b5e2-1c12a3c403f6/Animated-Logo_1.gif" width="150"/>
</div>

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
