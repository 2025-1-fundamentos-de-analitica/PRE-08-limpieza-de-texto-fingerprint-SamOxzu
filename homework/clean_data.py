#
# Referencia:
# https://openrefine.org/docs/technical-reference/clustering-in-depth
#

import nltk  # type: ignore
import pandas as pd  # type: ignore

def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame"""

    # Carga el archivo CSV leyendolo con pandas. Se recibe como
    # parámetro desde su llamada de main, llamada en el código
    # principal.
    df = pd.read_csv(input_file)
    return df

def create_normalized_key(df):
    """Cree una nueva columna en el DataFrame que contenga
    el key de la columna 'raw_text'"""

    # Genera una copia del DataFrame leído
    df = df.copy()

    # Copia la columna 'raw_text' a una nueva columna llamada 'key'
    df["key"] = df["raw_text"]

    # Remueve los espacios en blanco al principio y al final de la
    # cadena generada en la columna 'key'.
    df["key"] = df["key"].str.strip()

    # Por convención, debido a que en minería de datos los sustantivos
    # llevan mayúscula al principio, convertimos la cadena presente en
    # la columna 'key' a minúscula.
    df["key"] = df["key"].str.lower()

    # Debido a que es probable que existan palabras iguales con y sin guión,
    # eliminamos el guión de la cadena y lo remplazamos por un vacío.
    df["key"] = df["key"].str.replace("-", "")

    # Ya que los signos de puntuación no hacen parte de las palabras,
    # eliminamos los signos de puntuación de la cadena y los remplazamos
    # por un vacío. De manera similar, los caracteres especiales.
    df["key"] = df["key"].str.translate(
        str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    )    

    # Convertimos el texto ya filtrado, a una lista de tokens (palabras).
    df["key"] = df["key"].str.split()

    # Usando el algoritmo de stemming de Porter, reducimos cada token (palabra)
    # a su raíz, así logramos unificar palabras conjugadas o plurales.
    # Por ejemplo: "running" y "runner" se convierten en "run".
    stemmer = nltk.PorterStemmer()
    df["key"] = df["key"].apply(lambda x: [stemmer.stem(word) for word in x])

    # Mediante el uso de un conjunto set(), eliminamos los tokens duplicados
    # que se generaron en la columna 'key'. Y luego, los ordenamos de manera
    # ascendente.
    df["key"] = df["key"].apply(lambda x: sorted(set(x)))

    # Una vez que tenemos la lista de tokens únicos, los unimos nuevamente
    # en una cadena de texto separándolos por un espacio, para que se pueda
    # guardar en el archivo de salida.
    df["key"] = df["key"].str.join(" ")

    return df

def generate_cleaned_text(df):
    """Crea la columna 'cleaned_text' en el DataFrame"""

    keys = df.copy()

    # Ordenamos simultáneamente el dataframe por 'key' y 'text'.
    # para escoger el primer texto de cada grupo de 'key'.
    keys = keys.sort_values(by=["key", "raw_text"], ascending=[True, True])

    # Descartamos todas las filas duplicadas de la columna 'key' y
    # mantenemos la primera ocurrencia de cada clave (que obtuvimos
    # al ordenar el DataFrame).
    keys = df.drop_duplicates(subset="key", keep="first")

    # Creamos un diccionario con 'key' como clave y 'text' como valor
    # usando la función zip para combinar las dos columnas.
    key_dict = dict(zip(keys["key"], keys["raw_text"]))

    # Creamos una nueva columna 'cleaned_text' en el DataFrame original
    # y la llenamos con los valores del diccionario creado anteriormente.
    df["cleaned_text"] = df["key"].map(key_dict)

    return df

def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""

    # Crea una copia del DataFrame para evitar modificar el original
    df = df.copy()
    # Elimina la columna 'key' del DataFrame, solo seleccionando las
    # columnas 'raw_text' y 'cleaned_text'.
    df = df[["raw_text", "cleaned_text"]]
    # Guarda el DataFrame en un archivo CSV usando pandas.
    # Se recibe como parámetro desde su llamada de main, llamada
    # en el código principal.
    # Se usa el parámetro index=False para evitar que pandas
    # escriba el índice del DataFrame en el archivo CSV.
    df.to_csv(output_file, index=False)


def main(input_file, output_file):
    """Ejecuta la limpieza de datos"""
    # Llama a la función load_data para cargar el archivo de
    # entrada que se definió en el código principal.
    df = load_data(input_file)
    # Llama a la función create_normalized_key y le pasa como
    # parámetro el DataFrame que se retornó desde la función
    # load_data.
    df = create_normalized_key(df)
    # Llama a la función generate_cleaned_text y le pasa como
    # parámetro el DataFrame que se retornó desde la función
    # create_normalized_key.
    df = generate_cleaned_text(df)
    # Convierte el DataFrame a un archivo CSV a un archivo
    # intermedio para verificar el resultado.
    df.to_csv("files/test.csv", index=False)
    # Guarda verdaderamente el archivo de salida
    # usando la función save_data.
    save_data(df, output_file)   


if __name__ == "__main__":
    # Pasa los archivos de entrada y salida como argumentos de
    # la función main.
    main(
        input_file="files/input.txt",
        output_file="files/output.txt",
    )