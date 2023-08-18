import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams

load_dotenv()

model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

text = """
El vulcanólogo del Instituto Geofísico del Perú (IGP), José del Carpio, informó que la explosión se registró a las 5:30 am. y las emisiones se han dispersado con dirección a distritos de la provincia de arequipeña, como San Juan de Tarucani, Salinas, Polobaya, Chiguata, Pocsi y Quequeña donde se han registrado caídas leves de cenizas finas desde las 6:30 am. hasta las 10:00 am. aproximadamente.

Por su parte director del Observatorio Vulcanológico del Sur del IGP, Marco Rivera, explicó a El Comercio que las cenizas también se pudieron percibir en los distritos Paucarpata, Mariano Melgar, Miraflores, Alto Selva Alegre, José Luis Bustamante, Cercado, Yanahuara, Cerro Colorado en incluso de ha divisado en el Pedregal. Las pequeñas partículas, como polvillo, se mantuvieron suspendidas en la atmosfera, como una gran neblina por más de tres horas.

“No es común la caída de cenizas en la zona de Arequipa. Los vientos raramente se dirigen a esa dirección, generalmente van para el este, noreste y sureste. Los pobladores que viven cerca al volcán siempre deben tener sus equipos de protección para que puedan usarlo cuando se presenten las explosiones, no solo pobladores de Moquegua, sino también de Arequipa, porque el volcán es muy variable”, señaló Marco Rivera.
"""

text2 = """
La ciudad de Arequipa se encuentra a 65 kilómetros de distancia del volcán Ubinas. Pese a la observación de cenizas suspendidas en la atmosfera de la ciudad, este proceso eruptivo es catalogado de bajo a moderado, de menor intensidad al proceso de 2019. Rivera manifestó que hay días que el volcán está en calma y otros, que genera explosiones. No es constante. Tampoco se puede determinar cuánto durará este proceso eruptivo. En el 2005 duró hasta el 2009, más de tres años, mientras que en el 2019 fue poco más de dos meses.
"""

text3 = """
Nuevos escenarios
El meteorólogo del Servicio Nacional de Meteorología e Hidrología del Perú (Senamhi) José Luis Ticona, reveló a este medio que en esta época los vientos habituales son hacía el este, con dirección a Moquegua y Puno.

“Particularmente el día de hoy, de manera excepcional, en horas de la noche y madrugada el viento ha soplado del volcán hacía el oeste. Ha sido muy particular, un cambio solo por horas. El pronostico para los siguientes días es que se mantenga en su condición habitual, hacia el este, es decir hacia las localidades de Moquegua”, manifestó José Luis Ticona.

Otro escenario que no descarta el meteorólogo es que exista una remota posibilidad de que se presenten lluvias ácidas. Si el volcán Ubinas extiende su proceso eruptivo hasta el verano (a finales de diciembre, enero, febrero y marzo), podría toparse con la temporada de lluvias y producirse esta anomalía, aunque también señaló que, por causa del Fenómeno del Niño Global y el Niño Costero, se espera una deficiencia de lluvias.
"""

sentences = [text,text2, text3 ]

embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding, "\n")

qdrant_client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
)

def create_collection(collection_name):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vector_config=VectorParams(size=100, distance=Distance.COSINE))


