from LanguageModels import TinyLlama
from MediaMixing import StableDiffusionv1point5
from Preprocess import ImageCropperByYOLOv8

if __name__ == '__main__':
    prompt = 'design for me a cat with a lot of detail' # Your request

    # -> Languge Model
    LanguageModel = TinyLlama()    
    lm_output = LanguageModel.generate(prompt)
    print(f'lm_ouput: {lm_output}')

    # -> Addition Information
        # Image + Caption
        # Text

    image_path = 'None' # Your image path (None if not)
    addition_content = 'None' # Addition content (None if not)


    # -> Choose method for Media Mixing (1, 2)
    option = 1 # Method

    if (option == 1): # 1 -> Generative Approach
        MediaMixer = StableDiffusionv1point5() 
        image_output = MediaMixer.generate(lm_output=lm_output, 
                                        addition_content=addition_content, 
                                        image_path=image_path)
    elif (option == 2): # 2 -> Image Processing + Content synthesis
        ImageProcesser = ImageCropperByYOLOv8()
        ImageProcesser.cropping(option=1, image_path=image_path)
    
        