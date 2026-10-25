"""Run in Colab with publish(root, token). Never stores or prints the token."""
from pathlib import Path
import base64,json,urllib.request,urllib.error

REPOSITORY='jansenmisja/F1_dashboard'

def publish(root, token):
    root=Path(root)
    if not token.strip():raise ValueError('GitHub-token ontbreekt.')
    def api(method,path,payload=None,missing=False):
        req=urllib.request.Request('https://api.github.com/repos/'+REPOSITORY+('/'+path if path else ''),
            data=None if payload is None else json.dumps(payload).encode(),method=method,
            headers={'Authorization':'Bearer '+token.strip(),'Accept':'application/vnd.github+json',
                     'Content-Type':'application/json','User-Agent':'F1-dashboard-installer',
                     'X-GitHub-Api-Version':'2022-11-28'})
        try:
            with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
        except urllib.error.HTTPError as e:
            if missing and e.code in (404,409):return None
            raise RuntimeError(f'GitHub HTTP {e.code}; controleer repositoryrechten. Geen token in de foutmelding.') from None
        except (urllib.error.URLError,TimeoutError):
            raise RuntimeError('GitHub tijdelijk onbereikbaar. Controleer de repository voordat je opnieuw publiceert.') from None
    repository=api('GET','')
    if repository['full_name'].lower()!=REPOSITORY.lower():raise RuntimeError('Onverwachte repository.')
    ref=api('GET','git/ref/heads/main',missing=True)
    if ref is None:
        # Contents API can initialize an empty repository, unlike the Git tree API.
        api('PUT','contents/F1_HOSTING_INIT.md',{'message':'Initialize F1 hosting branch','branch':'main',
            'content':base64.b64encode(b'F1 Streamlit hosting\n').decode()})
        ref=api('GET','git/ref/heads/main')
    parent=ref['object']['sha']
    commit=api('GET','git/commits/'+parent)
    tree=api('GET','git/trees/'+commit['tree']['sha']+'?recursive=1')
    if tree.get('truncated'):raise RuntimeError('Repositoryboom te groot voor veilige installatie.')
    current={x['path']:x['sha'] for x in tree['tree'] if x['type']=='blob'}
    manifest=json.loads((root/'deployment_manifest.json').read_text())
    changes=[]
    import hashlib
    for name in manifest['publish_files']+['deployment_manifest.json']:
        path=root/name
        if not path.is_file() or path.is_symlink():raise RuntimeError('Installatiebestand ontbreekt: '+name)
        data=path.read_bytes()
        expected=manifest['sha256'].get(name)
        if expected and hashlib.sha256(data).hexdigest()!=expected:
            raise RuntimeError('Installatiebestand gewijzigd: '+name)
        blobsha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if current.get(name)==blobsha:continue
        blob=api('POST','git/blobs',{'content':base64.b64encode(data).decode(),'encoding':'base64'})
        changes.append({'path':name,'mode':'100644','type':'blob','sha':blob['sha']})
    if not changes:
        print('Repository is al bijgewerkt.');return
    newtree=api('POST','git/trees',{'base_tree':commit['tree']['sha'],'tree':changes})
    newcommit=api('POST','git/commits',{'message':'Deploy full public F1 workflow with durable project storage',
                                     'tree':newtree['sha'],'parents':[parent]})
    # force=False rejects concurrent branch updates instead of overwriting them.
    api('PATCH','git/refs/heads/main',{'sha':newcommit['sha'],'force':False})
    print('Bijgewerkt: https://github.com/'+REPOSITORY)
    print('Streamlit: branch main | bestand cloud_app.py | Python 3.13')
