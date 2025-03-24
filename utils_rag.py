import base64
from IPython.display import Image, display
import fitz
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image as Image2 
from langchain_core.documents import Document

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements, elements_to_json
import json

from collections import defaultdict

from pathlib import Path
import os
import time
from typing import List, Dict
from tqdm import tqdm

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import io

from cleaner_data import text_cleaner

def display_base64_image(base64_code):
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    # Display the image
    display(Image(data=image_data))


def data_general(chunks):
    # separate tables from texts
    tables = []
    texts = []
    images_b64 = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
        
    
    return tables, texts, images_b64



def plot_pdf_with_boxes(pdf_page, segments) :
  pix = pdf_page.get_pixmap()
  pil_image = Image2.frombytes ("RGB", [pix.width, pix.height], pix.samples)
  fig, ax = plt.subplots(1, figsize=(10, 10))
  ax. imshow (pil_image)
  categories = set()
  category_to_color = {
      'Title': 'blue',
      'Image': 'green',
      'Table': 'red',
  }
  for segment in segments:
    points = segment ["coordinates"] ["points"]
    layout_width = segment ["coordinates"] ["layout_width"]
    layout_height = segment ["coordinates" ] ["layout_height"]
    scaled_points = [
        (x * pix.width / layout_width, y * pix.height / layout_height)
        for x, y in points
        ]
    box_color = category_to_color.get(segment["category"], "deepskyblue")
    categories.add(segment["category"])
    rect = patches.Polygon(scaled_points, linewidth=1, edgecolor=box_color, facecolor="none")
    ax.add_patch(rect)

  # Make legend
  legend_handles = [patches.Patch(color="deepskyblue", label="Text" )]
  for category in ["Title", "Image", "Table"]:
    if category in categories:
      legend_handles.append(
          patches.Patch(color=category_to_color[category], label=category)
      )
  ax.axis("off")
  ax.legend(handles=legend_handles, loc="upper right")
  plt.tight_layout()
  plt.show()

def render_page(doc_list: list, page_number: int,file_path, print_text=True) -> None:

  pdf_page = fitz.open(file_path).load_page(page_number - 1)
  page_docs = [
      doc for doc in doc_list if doc.metadata.get ("page_number") == page_number
  ]
  segments = [doc.metadata for doc in page_docs]
  plot_pdf_with_boxes(pdf_page, segments)
  if print_text:
    for doc in page_docs:
      print(f"{doc. page_content}\n" )




def extract_page_numbers_from_chunks(chunk):
  
  elements = chunk.metadata.orig_elements
  page_numbers = set()

  for element in elements:
    page_numbers.add(element.metadata.page_number)
  
  return page_numbers

def display_chunk_pages(chunk,file_path):
  
  
  page_numbers = extract_page_numbers_from_chunks(chunk)
  
  docs = []

  for element in chunk.metadata.orig_elements:
    metadata = element.metadata.to_dict()
    if "Table" in str(type(element)):
      metadata["category"] = "Table"
    elif "Image" in str(type(element)):
      metadata["category"] = "Image"
    else:
      metadata["category"] = "Text"

    metadata["page_number"] = element.metadata.page_number

    docs.append(Document(page_content=element.text, metadata=metadata))

  for page_number in sorted(page_numbers):
    render_page(docs, page_number,file_path, False)


def ejecutar_chunking_pdf(dict_pdfs,ejecutar_pdf=False):
    
    if ejecutar_pdf:
        output_path = "data/"

        chunks = partition_pdf(
            filename=dict_pdfs['file_path'],
            infer_table_structure=True,            # extract tables
            strategy="hi_res",                     # mandatory to infer tables

            extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
            image_output_dir_path=output_path,   # if None, images and tables will saved in base64

            extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

            chunking_strategy="by_title",          # or 'basic'
            max_characters=10000,                  # defaults to 500
            combine_text_under_n_chars=2000,       # defaults to 0
            new_after_n_chars=6000,
        )

        json_data = elements_to_json(chunks)

        
        with open(dict_pdfs['output_path'], 'w', encoding='utf-8') as f:
            f.write(json_data)

    else:
        with open(dict_pdfs['output_path'], 'r', encoding='utf-8') as f:
            json_data = f.read()

        # Convertir la cadena JSON en una lista de diccionarios
        element_dicts = json.loads(json_data)

        # Verificar que element_dicts es una lista de diccionarios
        if isinstance(element_dicts, list) and all(isinstance(d, dict) for d in element_dicts):
            # Convertir los diccionarios en elementos utilizando dict_to_elements
            chunks = dict_to_elements(element_dicts)
        else:
            raise ValueError("El contenido del archivo JSON no es una lista de diccionarios válida.")

    return chunks




def extract_text_with_page_mapping(chunks):
    """
    Extrae únicamente el texto de los chunks y mapea cada fragmento al número de página correspondiente.

    Args:
        chunks: Lista de elementos generados por ejecutar_chunking_pdf.

    Returns:
        List[Dict]: Lista de diccionarios con 'text' y 'page_number'.
    """
    text_chunks = []
    tables_html_chunks = []
    images_b64_chunks = []
    if len(chunks) > 0:
        file_name_pdf = chunks[0].to_dict()['metadata']['filename']


    for chunk in chunks:
        # Verificar si el chunk es un CompositeElement
        text_chunk = ''
        if "CompositeElement" in str(type(chunk)):
            for elem in chunk.metadata.orig_elements:
                page_numbers = extract_page_numbers_from_chunks(chunk)
                # Ignorar elementos de tipo Table e Image
                if "Table" not in str(type(elem)) and "Image" not in str(type(elem)):
                    texto = elem.text.strip()
                    text_chunk += texto + '\n'
                elif "Table" in str(type(elem)):
                    # Si el elemento es una tabla o una imagen, extraer los números de página
                    #tables_html_chunks.append(elem.metadata.text_as_html)
                    tables_html_chunks.append({
                            'text_html': elem.metadata.text_as_html,
                            'page_number': page_numbers,
                            'filename': file_name_pdf

                        })
                elif "Image" in str(type(elem)):

                    images_b64_chunks.append(
                        {
                            'images_b64': elem.metadata.image_base64,
                            'page_number': page_numbers,
                            'filename': file_name_pdf
                        }
                        )

            text_chunks.append({
                            'text': text_chunk,
                            'page_number': page_numbers,
                            'filename': file_name_pdf
                        })
            
        else:
            # Si el chunk no es CompositeElement, asumir que es un texto simple
            texto = chunk.text.strip()
            if texto:
                numero_pagina = chunk.metadata.page_number
                text_chunks.append({
                    'text': texto,
                    'page_number': numero_pagina
                })

    return text_chunks, tables_html_chunks, images_b64_chunks


## funciones añadidas 20250315


def cargar_json(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        datos = json.load(archivo)
    return datos

def get_chapters(chunks_docs, chapters,name, save=False):
    
    out_name = f'data/chunks_docs/chapters_ids_{name}.json'

    if save:
        chapter_ids = {}

        for chunk in chunks_docs:
            elements_chunks=chunk.metadata.orig_elements
            for chapter in chapters:
                
                for element in elements_chunks:
                    element_aux = element.to_dict()
                        
                    if element_aux['text']==chapter and (element_aux['type']=='Title' or element_aux['type']=='ListItem'):
      
                        chapter_ids[chapter]={}
                        chapter_ids[chapter]['element_id']=element_aux['element_id']
                        chapter_ids[chapter]['page_number']=element_aux['metadata']['page_number']
                    #break

        with open(out_name, 'w', encoding='utf-8') as f:
            json.dump(chapter_ids, f, indent=4)
    else:
        with open(out_name, 'r', encoding='utf-8') as f:
            chapter_ids = json.load(f)
    return chapter_ids



def create_path_if_not_exists(path_str):
    """
    Crea una ruta si no existe.

    Args:
        path_str (str): La ruta que se desea crear.
    """
    path = Path(path_str)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Ruta creada: {path}")
    else:
        print(f"La ruta ya existe: {path}")


def extract_text_with_page_mapping_v2(chunks,chapters,chapter_ids,name_dir,min_page_number=1,save=False):
    """
    Extrae únicamente el texto de los chunks y mapea cada fragmento al número de página correspondiente.

    Args:
        chunks: Lista de elementos generados por ejecutar_chunking_pdf.

    Returns:
        List[Dict]: Lista de diccionarios con 'text' y 'page_number'.
    """
    if len(chunks) > 0:
            file_name_pdf = chunks[0].to_dict()['metadata']['filename']

    path_str = f'data/chunks_docs/{name_dir}'
    file_name_aux = file_name_pdf.split('.')[0]
    out_name = f'{path_str}/text_chunks_{file_name_aux}.json'

    
    if save:
        text_chunks = []
        tables_html_chunks = []
        images_b64_chunks = []

        map_chapter = ''
        #chapter_see = []+map_chapter
        

        

        for chunk in chunks:

            chunk_aux = chunk.to_dict()
            # Verificar si el chunk es un CompositeElement

            if chunk_aux['type']=="CompositeElement":
   
                
                for elem in chunk.metadata.orig_elements:
                    element_aux = elem.to_dict()

                    ###corregir solo si es title o list item
                    if (element_aux['type'] == "Title" or element_aux['type'] == "ListItem") and element_aux['text'] in chapters:
                        if element_aux['metadata']['page_number']>=min_page_number:
                            map_chapter = chapter_ids[element_aux['text']]['element_id']
                            
                            print(f"Mapeando Capítulo: {element_aux['text']} - Element_id: {map_chapter}")

                        

                    # Ignorar elementos de tipo Table e Image
                    if element_aux['type'] != "Table" and element_aux['type'] != "Image":
                        #texto = elem.text.strip()
                        #text_chunk += texto + '\n'


                        text_chunks.append({
                                    'text': element_aux['text'],
                                    'page_number': [element_aux['metadata']['page_number']],
                                    'filename': file_name_pdf,
                                    'chapter_id': map_chapter
                                })



                    elif element_aux['type'] == "Table":
                        
                        # Si el elemento es una tabla o una imagen, extraer los números de página
                        #tables_html_chunks.append(elem.metadata.text_as_html)
                        tables_html_chunks.append({
                                'text_html': element_aux['metadata']['text_as_html'],
                                'page_number': [element_aux['metadata']['page_number']],
                                'filename': file_name_pdf,
                                'chapter_id': map_chapter

                            })
                    elif "Image" in str(type(elem)):

                        images_b64_chunks.append(
                            {
                                'images_b64': element_aux['metadata']['image_base64'] ,
                                'page_number': [element_aux['metadata']['page_number']],
                                'filename': file_name_pdf,
                                'chapter_id': map_chapter
                            }
                            )
                
                
                
            else:
                # Si el chunk no es CompositeElement, asumir que es un texto simple
                texto = chunk_aux['text'].strip()
                if texto:
                    numero_pagina = chunk_aux['metadata']['page_number']
                    text_chunks.append({
                        'text': texto,
                        'page_number': [numero_pagina],
                        'filename': file_name_pdf,
                        'chapter_id': map_chapter
                    })
        
        # Diccionario auxiliar para agrupar por chapter_id
        grouped_data = defaultdict(lambda: {'text': '', 'page_number': set(), 'filename': '', 'chapter_id': ''})

        # Agrupar los textos por chapter_id
        for item in text_chunks:
            chapter_id = item['chapter_id']
            grouped_data[chapter_id]['text'] += item['text'] + "\n"
            grouped_data[chapter_id]['page_number'].add(item['page_number'][0])
            grouped_data[chapter_id]['filename'] = item['filename']
            grouped_data[chapter_id]['chapter_id'] = chapter_id

        # Convertir el diccionario agrupado en una lista de diccionarios
        result = [{'text': value['text'].strip(), 'page_number': sorted(value['page_number']), 'filename': value['filename'], 'chapter_id': value['chapter_id']} for value in grouped_data.values()]


        
        create_path_if_not_exists(path_str)
        
        with open(out_name, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
        
        with open(f'{path_str}/tables_html_chunks_{file_name_aux}.json', 'w', encoding='utf-8') as f:
            json.dump(tables_html_chunks, f, indent=4)
        
        with open(f'{path_str}/images_b64_chunks_{file_name_aux}.json', 'w', encoding='utf-8') as f:
            json.dump(images_b64_chunks, f, indent=4)
            
    else:
        

        with open(f'data/chunks_docs/{name_dir}/text_chunks_{file_name_aux}.json', 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        with open(f'data/chunks_docs/{name_dir}/tables_html_chunks_{file_name_aux}.json', 'r', encoding='utf-8') as f:
            tables_html_chunks = json.load(f)

        with open(f'data/chunks_docs/{name_dir}/images_b64_chunks_{file_name_aux}.json', 'r', encoding='utf-8') as f:
            images_b64_chunks = json.load(f)
            
        


    return result, tables_html_chunks, images_b64_chunks




def create_summary_chain(llm: ChatGoogleGenerativeAI):
    """Crea los componentes básicos de la cadena de resumen."""
    prompt_text = """
    Eres un asistente cuya tarea es resumir tablas y textos.
    Da un resumen conciso de la tabla o el texto.
    
    Responde solo con el resumen, sin comentarios adicionales.
    No comiences tu mensaje diciendo "Aquí hay un resumen" ni nada por el estilo.
    Solo da el resumen tal como está.
    
    Siempre el resumen debe ser entregado en español.

    Para resumir una tabla tienes que mencionar los datos que se encuentran en ella y ademas concluirla con un resumen.
    
    Fragmento de tabla o texto: {element}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    return {"element": lambda x: x} | prompt | llm | StrOutputParser()

def process_texts(text_chunks: List[Dict],
                 llm: ChatGoogleGenerativeAI, 
                 text_key: str,
                 name_dir: str,
                 save: bool = False,
                 use_batch: bool = False,
                 max_concurrency: int = 3,
                 ) -> List[Dict]:
    """
    Procesa fragmentos de texto de forma secuencial o en lotes utilizando chain.apply().
    
    Args:
        text_chunks: Lista de diccionarios que contienen texto y metadatos.
        llm: Modelo de lenguaje que se utilizará para el procesamiento.
        text_key: Clave para acceder al contenido de texto en los fragmentos.
        use_batch: Si es True, procesa utilizando apply(); si es False, procesa secuencialmente.
        max_concurrency: Número máximo de solicitudes concurrentes para el procesamiento por lotes.
    """
    summarize_chain = create_summary_chain(llm)
    results = []

    if text_key == 'text':
        texts = [text_cleaner.clean_text(chunk[text_key]) for chunk in text_chunks]
    else:
        texts = [chunk[text_key] for chunk in text_chunks]
    # Parámetros para el control de la tasa
    requests_per_minute = 10
    tokens_per_minute = 500000
    seconds_per_request = 60 / requests_per_minute
    token_count = 0

    path_str = f'data/chunks_docs/{name_dir}'
    file_name_pdf = text_chunks[0]['filename']
    file_name_aux = file_name_pdf.split('.')[0]
    if save:
        try:
            if use_batch:
                # Procesamiento por lotes
                for i in range(0, len(texts), max_concurrency):
                    batch = texts[i:i + max_concurrency]
                    
                    # Estimar el número de tokens en el lote
                    batch_token_count = sum(len(llm.get_token_ids(text)) for text in batch)  
                    
                    if token_count + batch_token_count > tokens_per_minute:
                        print('Limite Excedido, esperando 60 segundos')
                        time.sleep(60)  # Espera un minuto si se excede el límite de tokens
                        token_count = 0
                    
                    summaries = summarize_chain.batch(texts, {"max_concurrency": max_concurrency})
                    
                    for j, summary in enumerate(summaries):
                        results.append({
                            'summary': summary,
                            'page_number': text_chunks[i + j].get('page_number'),
                            'filename': text_chunks[i + j].get('filename'),
                            'chapter_id': text_chunks[i + j].get('chapter_id'),
                            'status': 'success'
                        })
                    
                    token_count += batch_token_count
                    time.sleep(seconds_per_request * len(batch))
            else:
                # Procesamiento secuencial
                for chunk in tqdm(text_chunks, desc="Procesando secuencialmente"):
                    try:
                        if text_key == 'text':
                            texto = text_cleaner.clean_text(chunk[text_key])
                        else:
                            texto = chunk[text_key]
                        
                        # Estimar el número de tokens en el texto
                        texto_token_count = len(llm.get_token_ids(texto))
                        
                        if token_count + texto_token_count > tokens_per_minute:
                            print('Limite Excedido, esperando 60 segundos')
                            time.sleep(60)  # Espera un minuto si se excede el límite de tokens
                            token_count = 0
                        
                        resumen = summarize_chain.invoke({"element": texto})
                        results.append({
                            'summary': resumen,
                            'page_number': chunk.get('page_number'),
                            'filename': chunk.get('filename'),
                            'chapter_id': chunk.get('chapter_id'),
                            'status': 'success'
                        })
                        
                        token_count += texto_token_count
                        
                        time.sleep(seconds_per_request)
                    except Exception as e:
                        results.append({
                            'summary': None,
                            'page_number': chunk.get('page_number'),
                            'filename': chunk.get('filename'),
                            'chapter_id': chunk.get('chapter_id'),
                            'status': f'error: {str(e)}'
                        })
                
                        
        except Exception as e:
            print(f"Error en el procesamiento: {str(e)}")
            return []
        
        print(f"Cantidad de tokens procesados: {token_count}")

    
        create_path_if_not_exists(path_str)
        if text_key == 'text':
                out_name = f'{path_str}/text_chunks_summary_{file_name_aux}.json'
                with open(out_name, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)

        elif text_key == 'text_html':
                out_name = f'{path_str}/tables_html_chunks_summary_{file_name_aux}.json'
                with open(out_name, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
    else:
        if text_key == 'text':
            with open(f'{path_str}/text_chunks_summary_{file_name_aux}.json', 'r', encoding='utf-8') as f:
                results = json.load(f)
        elif text_key == 'text_html':
            with open(f'{path_str}/tables_html_chunks_summary_{file_name_aux}.json', 'r', encoding='utf-8') as f:
                results = json.load(f)

    return results





def resize_base64_image(base64_string, size=(128,128)):

    img_data = base64.b64decode(base64_string)
    img = Image2.open(io.BytesIO(img_data))
    resized_img = img.resize(size,Image2.LANCZOS)
    buffeder = io.BytesIO()
    resized_img.save(buffeder, format=img.format)
    return base64.b64encode(buffeder.getvalue()).decode('utf-8')

def create_summary_chain_image(llm: ChatGoogleGenerativeAI):
    """Crea los componentes básicos de la cadena de resumen."""
    prompt_template = """Describe la imagen en detalle. Recuerda hacer una descripción resumida,
    concisa y rescatar los detalles más importantes. Recuerda que
    
    todo tiene que ser en español
    """
    messages = [
    (
        "user",
        [
            {"type": "text", "text": prompt_template},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image}"},
            },
        ],
    )
]

    prompt = ChatPromptTemplate.from_messages(messages)

    
    return prompt | llm | StrOutputParser()

def process_images(images_chunks: List[Dict],
                 llm: ChatGoogleGenerativeAI,
                 resize: tuple,
                 name_dir: str,
                 imagen_key: str,
                 save: bool = False) -> List[Dict]:
    """
    Procesa fragmentos de texto de forma secuencial o en lotes utilizando chain.apply().
    
    Args:
        images_chunks: Lista de diccionarios que contienen imagenes y metadatos.
        llm: Modelo de lenguaje que se utilizará para el procesamiento.
        imagen_key: Clave para acceder al contenido de imagen en los fragmentos.

    """
    summarize_chain = create_summary_chain_image(llm)
    results = []

    
    # Parámetros para el control de la tasa
    requests_per_minute = 10
    tokens_per_minute = 500000
    seconds_per_request = 60 / requests_per_minute
    token_count = 0
    try:
        for chunk in tqdm(images_chunks, desc="Procesando secuencialmente"):    
            try:
                imagen = chunk[imagen_key]

                compressed_image = resize_base64_image(imagen, resize)
                
                # Estimar el número de tokens en el texto
                texto_token_count = len(llm.get_token_ids(imagen))
                
                if token_count + texto_token_count > tokens_per_minute:
                    print('Limite Excedido, esperando 60 segundos')
                    time.sleep(60)  # Espera un minuto si se excede el límite de tokens
                    token_count = 0
                
                resumen = summarize_chain.invoke(compressed_image)
                results.append({
                    'summary': resumen,
                    'page_number': chunk.get('page_number'),
                    'filename': chunk.get('filename'),
                    'chapter_id': chunk.get('chapter_id'),
                    'status': 'success'
                })
                
                token_count += texto_token_count
                
                time.sleep(seconds_per_request)
            except Exception as e:
                results.append({
                    'summary': None,
                    'page_number': chunk.get('page_number'),
                    'filename': chunk.get('filename'),
                    'chapter_id': chunk.get('chapter_id'),
                    'status': f'error: {str(e)}'
                })
                        
    except Exception as e:
        print(f"Error en el procesamiento: {str(e)}")
        return []
    
    print(f"Cantidad de tokens procesados: {token_count}")

    path_str = f'data/chunks_docs/{name_dir}'
    file_name_pdf = images_chunks[0]['filename']
    file_name_aux = file_name_pdf.split('.')[0]
    if save:
        create_path_if_not_exists(path_str)
        
        out_name = f'{path_str}/images_chunks_summary_{file_name_aux}.json'
        with open(out_name, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

    else:
        with open(f'{path_str}/images_chunks_summary_{file_name_aux}.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        
    return results



### funciones añadidas 20250317

import os
import pickle
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
import json
import umap
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


# Funciones para persistir el docstore
def save_docstore(docstore, filename):
    with open(filename, "wb") as f:
        # Guardamos el diccionario interno del docstore
        pickle.dump(docstore.store, f, pickle.HIGHEST_PROTOCOL)

def load_docstore(filename):
    store = InMemoryStore()
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            store_data = pickle.load(f)
        # Se recarga el contenido en el InMemoryStore
        store.mset(list(store_data.items()))
    return store



def cargar_chunks_proc(file_name_pdf,name_dir):
    file_name_aux = file_name_pdf.split('.')[0]
    with open(f'data/chunks_docs/{name_dir}/text_chunks_{file_name_aux}.json', 'r', encoding='utf-8') as f:
            result = json.load(f)
    with open(f'data/chunks_docs/{name_dir}/tables_html_chunks_{file_name_aux}.json', 'r', encoding='utf-8') as f:
        tables_html_chunks = json.load(f)
    with open(f'data/chunks_docs/{name_dir}/images_b64_chunks_{file_name_aux}.json', 'r', encoding='utf-8') as f:
        images_b64_chunks = json.load(f)
    return result,tables_html_chunks,images_b64_chunks

def cargar_summary_proc(file_name_pdf,name_dir):
    file_name_aux = file_name_pdf.split('.')[0]
    with open(f'data/chunks_docs/{name_dir}/text_chunks_summary_{file_name_aux}.json', 'r', encoding='utf-8') as f:
        result = json.load(f)
    with open(f'data/chunks_docs/{name_dir}/tables_html_chunks_summary_{file_name_aux}.json', 'r', encoding='utf-8') as f:
        tables_html_chunks = json.load(f)
    with open(f'data/chunks_docs/{name_dir}/images_chunks_summary_{file_name_aux}.json', 'r', encoding='utf-8') as f:
        images_b64_chunks = json.load(f)
    return result,tables_html_chunks,images_b64_chunks


def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings


def query_to_embedding(query, retriever):
    query_embedding = retriever.vectorstore._embedding_function.embed_query(query)
    return query_embedding

def query_to_docs_embedding(original_query_embedding, retriever,n_results=5):
    
    results_emb = retriever.vectorstore._chroma_collection.query(
    query_embeddings=[original_query_embedding]
    ,n_results=n_results
    , include=['documents', 'embeddings']
    )
    retrieved_embeddings = results_emb['embeddings'][0]
    return retrieved_embeddings




def plot_umap_embeddings(umap_transform,projected_dataset_embeddings, retriever, query=None):

    original_query_embedding = query_to_embedding(query, retriever)
    retrieved_embeddings = query_to_docs_embedding(original_query_embedding, retriever)
    projected_original_query_embedding = project_embeddings([original_query_embedding], umap_transform)
    projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)


    # Plot the projected query and retrieved documents in the embedding space
    plt.figure()
    plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
    plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')
    plt.scatter(projected_original_query_embedding[:, 0], projected_original_query_embedding[:, 1], s=150, marker='X', color='r')


    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'{[query[0:60]]}...')
    plt.axis('off')

    return plt, projected_retrieved_embeddings, projected_original_query_embedding



def plot_multiple_pages(file_path: str, page_numbers: list) -> None:
    """
    Visualiza múltiples páginas de un PDF con sus cajas delimitadoras.
    
    Args:
        file_path (str): Ruta al archivo PDF
        page_numbers (list): Lista de números de página a visualizar
        
    Example:
        plot_multiple_pages("ruta/al/archivo.pdf", [1, 2, 5])
    """
    # Cargar el documento PDF
    pdf_doc = fitz.open(file_path)
    
    # Calcular el número de filas y columnas para el subplot
    n_pages = len(page_numbers)
    n_cols = min(3, n_pages)  # Máximo 3 columnas
    n_rows = (n_pages + n_cols - 1) // n_cols
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(7*n_cols, 10*n_rows))
    
    # Para cada página solicitada
    for idx, page_num in enumerate(sorted(page_numbers)):
        try:
            # Cargar la página
            pdf_page = pdf_doc.load_page(page_num - 1)  # Restamos 1 porque fitz usa índice base 0
            pix = pdf_page.get_pixmap()
            
            # Convertir a imagen PIL
            img = Image2.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Crear subplot
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            ax.imshow(img)
            ax.set_title(f'Página {page_num}')
            ax.axis('off')
            
        except Exception as e:
            print(f"Error al procesar página {page_num}: {str(e)}")
    
    # Ajustar espaciado entre subplots
    plt.tight_layout()
    plt.show()
    
    # Cerrar el documento PDF
    pdf_doc.close()

    return fig


def plot_multiple_pages_vertical(file_path: str, page_numbers: list) -> None:
    """
    Visualiza múltiples páginas de un PDF verticalmente, una debajo de otra.
    
    Args:
        file_path (str): Ruta al archivo PDF
        page_numbers (list): Lista de números de página a visualizar
    """
    # Cargar el documento PDF
    pdf_doc = fitz.open(file_path)
    
    # Para cada página solicitada
    for page_num in sorted(page_numbers):
        try:
            # Cargar la página
            pdf_page = pdf_doc.load_page(page_num - 1)
            pix = pdf_page.get_pixmap()
            
            # Convertir a imagen PIL
            img = Image2.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Crear una nueva figura para cada página
            plt.figure(figsize=(10, 14))
            plt.imshow(img)
            plt.title(f'Página {page_num}')
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Error al procesar página {page_num}: {str(e)}")
    
    # Cerrar el documento PDF
    pdf_doc.close()

def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)


from sentence_transformers import CrossEncoder

def retrieved_documents(query
                        , retriever
                        , n_results=5
                        , reranker=True
                        , model_name_reranker='BAAI/bge-reranker-v2-m3'
                        , path_data = "data/docs/"
                        , print_results=True):
    
    
    original_query_embedding = query_to_embedding(query, retriever)

    results_emb = retriever.vectorstore._chroma_collection.query(
        query_embeddings=[original_query_embedding]
        ,n_results=n_results
        , include=['documents', 'embeddings']
        )
    
    all_data_vector_store = retriever.vectorstore.get(include=['documents','metadatas'])
    id_to_doc_id = {id_ : (meta['doc_id'],meta['file_type']) for id_, meta in zip(all_data_vector_store['ids'], all_data_vector_store['metadatas'])}
    doc_to_id = dict(zip(all_data_vector_store['documents'], all_data_vector_store['ids']))
      
    if reranker:
        retrieved_documents = results_emb['documents'][0]

        pairs = [[query, doc] for doc in retrieved_documents]
        cross_encoder = CrossEncoder(model_name_reranker)
        scores = cross_encoder.predict(pairs)

        # Ordenar documentos por puntuación
        sorted_pairs = sorted(zip(scores, retrieved_documents), reverse=True)

        # lista de id de documentos
        id_docs = []
        chapter_id = []
        object_elements = []
        
        doc_orig_list = []
        # Ordenar documentos por puntuación
        sorted_pairs = sorted(zip(scores, retrieved_documents), reverse=True)

        # Imprimir documentos ordenados con sus puntuaciones
        
        for score, doc in sorted_pairs:

            id_encontrado = doc_to_id.get(doc, 'No encontrado')
            id_docs.append(id_encontrado)
            doc_id_encontrado = id_to_doc_id.get(id_encontrado,'No encontrado')
            
            doc_orig=retriever.docstore.mget([doc_id_encontrado[0]])[0]
            
            doc_orig_list.append(doc_orig)

            chapter_id.append(doc_orig['chapter_id'])
            object_elements.append(doc_id_encontrado[1])
            file_name_aux = path_data + doc_orig['filename']
            if print_results:
                print("\n=== Documentos Ordenados por Relevancia ===")
                print("="*100)
                print(f"\nPuntuación: {score:.4f}")
                print(f"Contenido resumido:\n {word_wrap(doc,n_chars=200)}")
                print(doc_id_encontrado)
                try:
                    print(f"Contenido original:\n {doc_orig['text']}")
                except KeyError:
                    try:
                        print(f"Contenido original:\n {doc_orig['text_html']}")
                    except KeyError:
                        try:
                            print(f"Contenido original:\n {doc_orig['images_b64']}")
                        except KeyError:
                            print("No se encontró el contenido en ningún formato conocido")
            
                print(f"Paginas: {doc_orig['page_number']}")
                print(f"Nombre archivo: {doc_orig['filename']}")
                print(f"Capitulo: {doc_orig['chapter_id']}")
                plot_multiple_pages(file_name_aux, doc_orig['page_number'])
                #lot_multiple_pages_vertical(file_name_aux, doc_orig['page_number'])

        return sorted_pairs, id_docs, chapter_id, object_elements, doc_orig_list
    else:
        # lista de id de documentos
        
        id_docs = []
        chapter_id = []
        object_elements = []
        doc_orig_list = []
        doc_summary=[]
        for doc in results_emb['documents'][0]:
            doc_summary.append(doc)
            id_encontrado = doc_to_id.get(doc, 'No encontrado')
            id_docs.append(id_encontrado)
            doc_id_encontrado = id_to_doc_id.get(id_encontrado,'No encontrado')
            doc_orig=retriever.docstore.mget([doc_id_encontrado[0]])[0]

            doc_orig_list.append(doc_orig)
 
            chapter_id.append(doc_orig['chapter_id'])
            object_elements.append(doc_id_encontrado[1])
            file_name_aux = path_data + doc_orig['filename']
            if print_results:
                print("\n=== Documentos sin Relevancia ===")
                print("="*100)
                print(f"Contenido resumido:\n {word_wrap(doc,n_chars=200)}")
                print(doc_id_encontrado)
                try:
                    print(f"Contenido original:\n {doc_orig['text']}")
                except KeyError:
                    try:
                        print(f"Contenido original:\n {doc_orig['text_html']}")
                    except KeyError:
                        try:
                            print(f"Contenido original:\n {doc_orig['images_b64']}")
                        except KeyError:
                            print("No se encontró el contenido en ningún formato conocido")
                
                print(f"Paginas: {doc_orig['page_number']}")
                print(f"Nombre archivo: {doc_orig['filename']}")
                print(f"Capitulo: {doc_orig['chapter_id']}")
                plot_multiple_pages(file_name_aux, doc_orig['page_number'])
                #plot_multiple_pages_vertical(file_name_aux, doc_orig['page_number'])
        return doc_summary, id_docs, chapter_id, object_elements, doc_orig_list
    


### funiones añadidas 20250321

def generar_citas(a_list, d_list, e_list,use_reranker=False):
    resultado = []
    for a_eval,d_val, e_val in zip(a_list,d_list, e_list):
        cita = {}
        if use_reranker:
            if d_val == 'Imagen':
                # Buscar el primer valor de a_list que coincide con d_val
                contenido = a_eval[1]
            elif d_val == 'Table':
                contenido = a_eval[1]
            else:
                contenido = e_val.get('text', '')
        else:
            if d_val == 'Imagen':
                # Buscar el primer valor de a_list que coincide con d_val
                contenido = a_eval
            elif d_val == 'Table':
                contenido = a_eval
            else:
                contenido = e_val.get('text', '')


        page_number = e_val.get('page_number', '')
        filename = e_val.get('filename', '')


        if contenido is not None:
            
            cita['contenido'] = contenido
            cita['page_number'] = page_number
            cita['filename'] = filename
            cita['file_type'] = d_val
            resultado.append(cita)
    
    contexto = "============================== Ventana de Contexto=======================================\n\n"
    for i,cita in enumerate(resultado):
        contexto += f"----------------------------Inicio Contexto {i+1}--------------------------------\n\n"
        contexto += f"Página: {cita['page_number']}\n\n"
        contexto += f"Archivo: {cita['filename']}\n\n"
        contexto += f"Tipo: {cita['file_type']}\n\n"
        contexto += f"Contenido: {cita['contenido']}\n\n"
        
        contexto += f"----------------------------Fin Contexto {i+1}--------------------------------\n\n"
    contexto += "==============================Fin Ventana de Contexto =======================================\n\n"

    
    return resultado, contexto


def estimar_costo(response, model):
    cost_resp=response.usage_metadata

    precios_millon= {
        'gemini-2.0-flash':{'entrada':0.1, 'salida':0.4},
        'gemini-1.5-flash':{'entrada':0.075, 'salida':0.3},
        'gemini-1.5-pro':{'entrada':0.15, 'salida':0.6},
    }

    if model not in precios_millon:
        raise ValueError(f"Modelo no encontrado: {model}, modelos disponibles: {precios_millon.keys()}")
    
    tokens_entrada = cost_resp['input_tokens']
    tokens_salida = cost_resp['output_tokens']

    costo_entrada = precios_millon[model]['entrada'] * tokens_entrada / 1e6
    costo_salida = precios_millon[model]['salida'] * tokens_salida / 1e6

    costo_total = costo_entrada + costo_salida
    return costo_total
