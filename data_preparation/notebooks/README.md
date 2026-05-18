# Insula API and Insula AI Jupyter Examples

A set of Jupyter Notebooks showing the usage of Insula's APIs, in particular:

- [Insula Access & Discovery](https://github.com/cgi-italy/notebooks/blob/main/InsulaAccess&Discovery.ipynb): query collections and download files
- [Insula SearchAPI](https://github.com/cgi-italy/notebooks/blob/main/InsulaSearchAPI.ipynb): catalog search, download api and WMS/WFS capabilities
- [Insula Data Collections](https://github.com/cgi-italy/notebooks/blob/main/InsulaDataCollections.ipynb): create collections and upload files
- [Insula Processing Services](https://github.com/cgi-italy/notebooks/blob/main/InsulaProcessingServices.ipynb): see available services and launch one.
This notebook requires to create a demo processor, follow [these instructions](https://github.com/cgi-italy/notebooks/blob/main/SimpleDemoProcessor) to do it
- [Insula OpenEO](https://github.com/cgi-italy/notebooks/blob/main/InsulaOpenEo.ipynb): exploit Insula's OpenEO backend and capabilities (ECOSTRESS dataset).
- [Copernicus OpenEO](https://github.com/cgi-italy/notebooks/blob/main/CopernicusOpenEO.ipynb): learn how to use the OpenEO library and services to access external OpenEO datasets.
- [Kubeflow Pipeline](https://github.com/cgi-italy/notebooks/blob/main/KubeflowHelloWorld.ipynb): Jupyter Notebook to produce yaml file of a simple "Hello world!" pipeline
- [Collection Batch Download Script](https://github.com/cgi-italy/notebooks/blob/main/CollectionBatchDownloadScript): A fully standalone python script of what can be achieved by leveraging Insula APIs. A simple yet often requested use-case that allows us to automatically download all contents from a collection with many files.

## Requirements
Each notebook has a line containing a `pip install` command with only the packages required to run it. If you prefer to install them all they are listed inside the file `requirements.txt`:
```bash
pip install -r requirements.txt
```
 
## Create a Virtual Environment inside Jupyter 
If you want these packages to survive Jupyter server reboots a virtual environment can be created. To do it open a terminal inside Jupyter and use the following commands:

```bash
# Create a new environment:
python -m venv /home/jovyan/InsulaNotebookEnv

# Make it active:
source /home/jovyan/InsulaNotebookEnv/bin/activate

# Install the required packages:
pip install -r requirements.txt

# Enable the environment for usage in a notebook’s kernel
pip install ipykernel
ipython kernel install --user --name=InsulaNotebookEnv
```

## How to switch Insula instance
The first block will always be the one which generates the authorization and connects to the desired endpoint. Just change the values with the ones below to use it on another instance:


- ESA-MAAP:
```python
BASE_URL="https://biomass.pal.maap.eo.esa.int"
insulaAuth: InsulaOpenIDConnect = InsulaOpenIDConnect(
        authorization_endpoint="https://identity.pal.maap.eo.esa.int/realms/biomass/protocol/openid-connect/auth",
        token_endpoint="https://identity.pal.maap.eo.esa.int/realms/biomass/protocol/openid-connect/token",
        redirect_uri="http://localhost:9207/auth",
        client_id="api-client"
    )
```
- DESP:
```python
BASE_URL="https://insula.e2e-2.desp.space"
insulaAuth: InsulaOpenIDConnect = InsulaOpenIDConnect(
        authorization_endpoint="https://iam.e2e-2.desp.space/realms/desp/protocol/openid-connect/auth",
        token_endpoint="https://iam.e2e-2.desp.space/realms/desp/protocol/openid-connect/token",
        redirect_uri="https://broker.datawkfl.e2e-2.desp.space",
        client_id="hda-broker-public"
    )
```
- FS-TEP:
```python
BASE_URL="https://foodsecurity-explorer.insula.earth/"
insulaAuth: InsulaOpenIDConnect = InsulaOpenIDConnect(
        authorization_endpoint="https://identity.insula.earth/realms/foodsecurity-tep/protocol/openid-connect/auth",
        token_endpoint="https://identity.insula.earth/realms/foodsecurity-tep/protocol/openid-connect/token",
        redirect_uri="http://localhost:9207/auth",
        client_id="api-client"
    )
```
- PHISAT2:
Remember that the Phisat2 instance requires a mail verification.
Before using the API be sure to log-in via ui at least once and complete the verification process.
```python
BASE_URL="https://phisat2.insula.earth"
insulaAuth: InsulaOpenIDConnect = InsulaOpenIDConnect(
        authorization_endpoint="https://identity.insula.earth/realms/phisat2/protocol/openid-connect/auth",
        token_endpoint="https://identity.insula.earth/realms/phisat2/protocol/openid-connect/token",
        redirect_uri="http://localhost:9207/auth",
        client_id="api-client"
    )
```

## PhiSat2 Collections
Collections I have access through API.
| ID | Name                        | Number of files |
| -- | --------------------------- | --------------- |
| 7  | Phisat2-L1                  | 2000+           |
| 21 | Phisat2-L2-CloudDetection   | 6               |
| 22 | Phisat2-L2-Sat2Map          | 0               |
| 23 | Phisat2-L2-VesselDetection  | 0               |
| 24 | Phisat2-L2-DeepCompression  | 0               |
| 25 | Phisat2-L2-PhiFireAI        | 0               |
| 26 | Phisat2-L2-MarineProtection | 0               |

Changed nd collection_id in `insula_download_collection.py` to 7 and parameters in `config.py`:
```python
class Download:
    authorization_endpoint="https://identity.insula.earth/realms/phisat2/protocol/openid-connect/auth"
    token_endpoint="https://identity.insula.earth/realms/phisat2/protocol/openid-connect/token"
    redirect_uri="http://localhost:9207/auth"
    client_id="api-client"
    base_url="https://phisat2.insula.earth"
```