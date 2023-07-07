import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle
import os

# Carregue o modelo a partir do arquivo pickle
model_path = "../models/toxic_to_pet.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

def predict(imagem):
    # Salva a imagem em um arquivo temporário
    temp_file = "temp.png"
    imagem.save(temp_file)
    
    img = image.load_img(temp_file, target_size=(96, 96))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    class_index = np.argmax(classes)
    class_names = ['cebola', 'alho', 'uva', 'passas', 'macadamia', 'massa_pao', 'lirio', 'azaleia', 'aloe_vera', 'filodendros', 'dieffenbachia', 'begonia', 'ciclamen', 'hortensia', 'comigo_ninguem_pode', 'coroa_de_cristo', 'abacate']
    inferred_class = class_names[class_index]
    
    # Remove o arquivo temporário
    os.remove(temp_file)
    
    return inferred_class, class_index


def main():
    st.title("Classificador de imagens de plantas e alimentos tóxicos para pets")
    
    # Exemplo de upload de imagem
    uploaded_file = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Abre a imagem
        imagem = Image.open(uploaded_file)
        
        # Exibe a imagem
        st.image(imagem, caption="Imagem carregada")
        
        # Faz a previsão
        predicted_class = predict(imagem)
        
        # Exibe a classe inferida
        st.write("Classe inferida:", predicted_class[0])
        st.write("Score:", predicted_class[1])

if __name__ == "__main__":
    main()
