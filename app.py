from flask import Flask, request, render_template
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import util
from gpt4all import GPT4All
import torch
import ast
import pandas as pd


para_df=pd.read_csv('data/paragraphs_embeddings.csv')
embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
def order_documents(query):
    documents=para_df['page_content'].tolist()
    Embeddings=para_df['Embeddings'].tolist()
    output_list = [ast.literal_eval(item) for item in Embeddings]
    query_result = embeddings.embed_query(query)
    query_result = torch.Tensor(query_result)
    Embeddings1 = torch.Tensor(output_list)
    hits = util.semantic_search(query_result, Embeddings1, top_k=5)[0]
    docs = []
    for hit in hits:
        if hit['score']>=0.5:
            document = documents[hit['corpus_id']]
            print("(Score:{:.4f})".format(hit['score']))
            # print(document)
            docs.append(document) 
    docs = ' '.join(docs) 
    return docs


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    comment = [x for x in request.form.values()]
    print(comment[0])
    query = comment[0]
    res=order_documents(query)
    print(res)
    if res == "":
        output="I don't know"
    else:
        prompt1 = f"""text:{res}


                Answer the below Question only from the text above.If the answer cannot be found from text above respond like 'I don't know'.

                Question:{query}
                Answer:"""
        output = model.generate(prompt=prompt1,temp=0, max_tokens=512)
        print(output)
    return render_template('index.html', prog=output)
    


if __name__ == '__main__':
    model = GPT4All(model_name='orca-mini-3b-gguf2-q4_0.gguf')
    app.run(debug=True)

