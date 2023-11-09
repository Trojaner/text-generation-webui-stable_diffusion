# Stable Diffusion Extension for text-generation-webui
Integrates image generation functionality to [text-generation-webui](https://github.com/oobabooga/text-generation-webui) using Stable Diffusion.  
Requires stable-diffusion-webui with enabled API.

Demo: 
<p align="left">
  <img src="/assets/demo1.png" width="33%" />
  <img src="/assets/demo2.png" width="33%" /> 
</p>

> **Note**
> This extension has been only tested with the [SD.Next](https://github.com/vladmandic/automatic) fork of stable-diffusion-webui but may still work with the vanilla [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) build as well.

## Features
- Generate images using stable-diffusion-webui.
- Well documented [settings](https://github.com/Trojaner/text-generation-webui-stable_diffusion/blob/main/settings.debug.yaml) file for easy configuration.
- Supports face swapping for generating consistent character images without needing loras. See [Ethical Guidelines](#ethical-guidelines) for more information.
- Multi-threading support - can handle concurrent chat sessions and requests.

## Supported Stable Diffusion WebUI Extensions
- [FaceSwapLab](https://github.com/glucauze/sd-webui-faceswaplab)

## Installation
- Open command prompt / terminal inside the text-generation-webui directory or navigate to it with `cd`. 
- Run `git clone https://github.com/Trojaner/text-generation-webui-stable_diffusion extensions/stable_diffusion`.
- Open the `settings.debug.yaml` file and adjust the settings to your liking.

> **Note**
> If you install this extension manually, make sure the extension directory is called stable_diffusion. 

## Development Environment Setup

**Pre-requisites**  
text-generation-webui, Visual Studio Code and Python 3.10 are required for development.  

**Setting up Visual Studio Code for development**
- [Install the extension first](#installation) if you haven't already.
- Start Visual Studio Code and open the stable_diffusion directory, then trust the repository if it asks you for it.
- Install the [recommended extensions](https://github.com/Trojaner/text-generation-webui-stable_diffusion/blob/main/.vscode/extensions.json) as they are required for code completion, linting and auto formatting.
- Adjust `.vscode/launch.json` to use your preferred model for debugging or install the default model [mistral-7b-instruct-v0.1.Q5_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/blob/main/mistral-7b-v0.1.Q5_K_M.gguf) instead.
- Once you want to test your changes, hit F5 (*Start Debugging*) to debug text-generation-webui with this extension pre-installed and with the `settings.debug.yaml` file as the settings file. You can also use Ctrl + Shift + F5 (*Restart Debugging*) to apply any changes you made to the code by restarting the server from scratch. Checkout [Key Bindings for Visual Studio Code](https://code.visualstudio.com/docs/getstarted/keybindings) for more shortcuts.  
- Be sure to check out the [Contribution Guidelines](#contribution-guidelines) below before submitting a pull request.

## Contribution Guidelines
- This project relies heavily on type hints, please make sure to add them to your code as well or your pull request will likely get rejected.
- Always reformat your code using [Black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort) before committing (it should already do so when saving files if you have installed the recommended extensions).
- Make sure that both, [mypy](https://github.com/python/mypy) and [flake8](https://github.com/PyCQA/flake8), are not outputting any linting errors.
- Prefix local functions and variables with an underscore (`_`) to indicate that they are not meant to be used outside of the current file
- Use snake case when naming functions and variables, pascal case when naming classes and uppercase for constants.
- Do not use abbreviations for variable names (such as `ctx` instead of `context`) unless they are simple and common like `i` for index or `n` for number.
- Always document and include new parameters in the `settings.debug.yaml` file.
- Last but not least, ensure that you do not accidentally commit changes you might have made to the `settings.debug.yaml` or `launch.json` files unless intentional.

## Ethical Guidelines
This extension integrates with the FaceSwapLab extension for stable-diffusion-webui and hence allows to swap faces in the generated images. This extension is not intended to for the creation of non-consensual deepfake content. Please use this extension responsibly and do not use it to create such content. The main purpose of the face swapping functionality is to allow the creation of consistent images of text-generation-webui characters. If you are unsure whether your use case is ethical, please refrain from using this extension.

The maintainers and contributors of this extension cannot be held liable for any misuse of this extension but will try to prevent such misuse by all means.

## Todo
- Some basic Gradio UI for fine-tuning the extension parameters at runtime
- Character specific parameters
- Custom LogitsProcessor or grammar implementation for generating proper and weighted SD image generation prompts
- Support [ReActor](https://github.com/Gourieff/sd-webui-reactor) as alternative faceswap integration [[api implementation](https://github.com/Gourieff/sd-webui-reactor/blob/main/scripts/reactor_api.py)] 
- Test with vanilla AUTOMATIC1111
- Standalone mode using diffusers and without stable-diffusion-webui
- Integrate with other SD extensions / scripts?

## See also
- [sd_api_pictures](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/sd_api_pictures) - the original stable-diffusion-webui extension which inspired this one
- [sd_api_pictures_tag_injection](https://github.com/GuizzyQC/sd_api_pictures_tag_injection) - a fork of sd_api_pictures with tag injection support

## License
[MIT](https://github.com/Trojaner/text-generation-webui-stable_diffusion/blob/main/LICENSE)
