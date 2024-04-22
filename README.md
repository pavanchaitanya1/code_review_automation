# code_review_automation

Github App for Code Review Automation using LLM and RAG


------

create a python virtual env

----------

python libraries that need to be installed are below, using pip3 install <library-name> 

llama_index
langchain
llama_index-llms-ollama
llama_index-embeddings-langchain
llama-index-vector-stores-weaviate
llama-index-embeddings-huggingface
weaviate-client
flask
PyGithub

------

Install Ollama application and run it

--------

Install Docker application

--------

Setup, run below command in code_review_automation folder

``` pip install -e . ```

-----

Setup Weaviate Vector DB locally

go to the scripts folder and run below command

docker compose up -d

check in the docker application if the container is up and running

---------

Go to src folder and run

python3 create_index.py

python3 main.py


------------

Flask running

once flask is installed, run 

``` python3 app.py ```

-----------

SocketXP setup (for public url)

https://www.socketxp.com/iot/remote-access-python-flask-app-from-internet/

Once it is setup, you need to below steps for setting up local server

``` sudo socketxp login <auth0-key> ```
``` sudo socketxp connect <ip-from-flask> ```

