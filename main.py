from LanguageModels import TinyLlama
from MediaMixing import StableDiffusionv1point5

if __name__ == '__main__':
    prompt = input('Enter your request: ')

    # -> Languge Model
    LanguageModel = TinyLlama()    
    lm_output = LanguageModel.generate(prompt)

    # -> Addition Information
        # Image + Caption
        # Text
    image_path = input('Enter your image path (None if not): ')
    addition_content = input('Enter your addition content (None if not): ')

    # -> Choose method for Media Mixing
    option = input('Enter your method: ')

    if (option == 1): # 1 -> Generative Approach
        MediaMixer = StableDiffusionv1point5() 
        image_output = MediaMixer.generate(lm_output=lm_output, 
                                        addition_content=addition_content, 
                                        image_path=image_path)
    elif (option == 2): # 2 -> Image Processing + Content synthesis
        pass
        