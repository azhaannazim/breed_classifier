from fastai.vision.all import *
import gradio as gr
import timm

learn = load_learner('Breedmodel.pkl')

categories = learn.dls.vocab

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.Image(width=192, height=192)
label = gr.Label()
examples = ['bd.jpg']

intf = gr.Interface(fn=classify_image, inputs=image,
                    outputs=label, examples=examples)
intf.launch(inline=False, share=True)
