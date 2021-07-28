import os
import sys
import time
from cosmic.ext.ptypy.io import interaction
from cosmic.ext.ptypy.utils import verbose, parallel
import time
verbose.set_level(3)
from cosmic.preprocess.process_5321_class import Processor, IAServerDealer, DEFAULT

from authlib.integrations.requests_client import OAuth2Session
from authlib.oauth2.rfc7523 import PrivateKeyJWT

import urllib
import urllib.request
import urllib.parse
import json
import time

API_URL="https://api.nersc.gov/api/v1.2"
TOKEN_URL = "https://oidc.nersc.gov/c2id/token"

jwt = None
client_id = "w47zp3shjaps6"
private_key = '{"kty": "RSA", "n": "ruFgg_cusCyQc_O5ddP_mjBx8P6N3rYg-e05MOBNbIrhWSonYnhO2QiQlwKhXRyVpcDUmjJ_b7hwa_szRbGJQCOLO2MsiFhDvI6AgmQj7U0gdiKZ7oKE6HR23TlQpZw45nQibMt6bhBvB3y_4mMP2EJLrvy5jAqpbItqQ9FsecZcvMlgEx4CWRxUNdG9wCZ8ieOEZBzF1ErFdJMOzjo9-MacTtbJWRb_D9qvV0fnuAa27fcFvC90E8PFknyvAs7q46P5UeQvxM2ystUlzcQq6quNDKhII0lE4q1cRibr_BPMEVZtAdJkXEyLkCHbBo8A5XiV-sD5OcuPgd54intNAw", "e": "AQAB", "d": "hXSu7-ZJdd58WlBrfrTLAYLo4Q2RfJ0mqzPSii8SRrvxXtcheS7wlQXJOcSjGeh_dx-h3w6cW8i32l-37_6dDBpT3X1AdchN4O4qudbr5-MM27pqGC74eGCweQCNP-TpM0z7HGVnx-i4olEcKgqJA_MLyL1KZ8mXI6N888Y07vQjVtNa7iR63NrIMGb6YiSRfdpXh3W7pM8rRDnA2kWiGvSgLpqau6-mISFnw_ilVnzVgrhjRKo_IdSllQdD0D2904ZwWXEfkiYAt8WjeHKgW62S7i-tnvZezjHTRs_AcL7ar4O_d09zsicq1swqfGzz6jyTtXGnvojcHKgLWFwCKQ", "p": "30chiLbrkP2Zw9-E_vbyxnvrxi-drGeKnT0g1TKcyARXLOtKGYQGKIGGpSJbbEpx1vNqQt5bj0LJnwQiW8uLcICxMnb6mppMti3vSGfidi3EbFjTwQS5mP25bdIWOnKp9jNEpwuBSa48WiVy18SetboeIknx_sevMlnBzTWEF50", "q": "yIJ8DWKZMJgnF80_HtZPp-ToGS9wGeCp8RI2f__rBFPdMFyMXiWm8vsnUD-pHm5ve6kCjVSAXQCU979cGNB7pRcX6yG6D5Z9TogXExEYdM8tRVfW52NV8G4w_qkztDN4Lkuy6LpWL-Dck4_xQdCeiqe9VatoDReS5MNQea3B5R8", "dp": "Hdnqt9aM5BOjzTZDF7t4deT3fsW69OPa-m7Dxv0_TNaXuR-0BnlKQXwfvlA7nNzPH4fnuwzzfNHXFvV8in1KJT5vcmnJ04Wxn-HAThPliRtRWZL-rJ2vGq9BbVdbNXFDG_F6ykKVhH5Q_1RmaEaXWYCKqtbsDb4wxDCP4pm3EVk", "dq": "OBI9TKTw_-TcscxExYPa_LGHsltQtvvbqj2UnhDcEPa-2SJYYo-W80YGxWs4CPmLFYK64vjpeJiMEAgkYhATM6SUnL1uwaMs4YQvJ7bVJv00xLp-r3BY_QZjjyOUAOWPyyqCGpDZP0RbiqxrJCOJ4m5sBQQM8fRQsMZpv802drM", "qi": "Sgp7MAEy63WnxKttyz5nL8b6EajRck9TcQHYW0xBl3vq270qn4tn56rutWv---aSL4wWRhStrSDZqgHPaqi9quy26DYQc7FACdT6WemyVhJNRRY9oUz49N939i641lPFlxC6kJqy-CnH0hwfaA1GS11YkL1JTXQg1YQ_fLZ3SK4"}'

def api(path, data=None, as_form=False):
    """Make a superfacility api call.
    Path is a relative path w/o the leading /.
    Data (optional) is a python dictionary."""
    global jwt

    url = "%s/%s" % (API_URL, path)
    if data:
        if as_form:
            encoded_data = urllib.parse.urlencode(data).encode()
            req =  urllib.request.Request(url, data=encoded_data)
            req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
            req.add_header("Content-Length", len(encoded_data))
            if jwt:
                req.add_header("Authorization", jwt)
            response = urllib.request.urlopen(req)
        else:
            jsondata = json.dumps(data)
            jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
            req = urllib.request.Request(url)
            req.add_header("Content-Type", "application/json")
            if jwt:
                req.add_header("Authorization", jwt)
            response = urllib.request.urlopen(req, jsondataasbytes)
    else:
        req = urllib.request.Request(url)
        if jwt:
            req.add_header("Authorization", jwt)
        response = urllib.request.urlopen(req)
    return json.load(response)

def get_new_jwt():
    global jwt, private_key
    if not private_key:
        private_key = input("Me want crendetials").strip()
    client = OAuth2Session(client_id=client_id,
                            client_secret=private_key,
                            token_endpoint_auth_method="private_key_jwt")
    client.register_client_auth_method(PrivateKeyJWT(TOKEN_URL))
    resp = client.fetch_token(TOKEN_URL, grant_type="client_credentials")
    jwt = resp["access_token"]

get_new_jwt()

api_job = """#!/bin/bash -l

#SBATCH -q realtime
#SBATCH -A als
#SBATCH -N 1
#SBATCH -C haswell
##SBATCH --tasks-per-node=32
#SBATCH -t 00:10:00
##SBATCH -J apidemo
#SBATCH --sdn
#SBATCH --exclusive

export HDF5_USE_FILE_LOCKING=FALSE
#echo SDN_IP=$SDN_IP_ADDR #doesn't work before srun
module load python
source activate ptypy_on_lazy_mpi4py
#export PYTHONPATH=/global/homes/b/benders/code/ptypy-rc/build/lib

srun -n 10 python ~/cosmic_stream_demo/stream_load_and_run.py %s

"""

use_globus = True

args = sys.argv[1:]

if len(args) > 0:
    param_json = args[0]
    import json
    # maybe restrict to master process here
    f = open(param_json, 'r')
    pars = json.load(f)
    f.close()
else:
    pars = {}

if len(args) > 1:
    host, port = args[1:]
    port = int(port)
else:
    host, port = "tcp://127.0.0.1", 5560

if parallel.master:
    S = interaction.Server(
        address=host,
        port=port,
    )
    S.activate()
else:
    S = None


## Globus
if use_globus and parallel.master:

    TOKEN_FILE='refresh-tokens.json'
    def load_tokens_from_file(filepath):
        """Load a set of saved tokens."""
        with open(filepath, "r") as f:
            tokens = json.load(f)

        return tokens


    def save_tokens_to_file(filepath, tokens):
        """Save a set of tokens for later use."""
        with open(filepath, "w") as f:
            json.dump(tokens, f)


    def update_tokens_file_on_refresh(token_response):
        """
        Callback function passed into the RefreshTokenAuthorizer.
        Will be invoked any time a new access token is fetched.
        """
        save_tokens_to_file(TOKEN_FILE, token_response.by_resource_server)


    def activate_endpoint(ep_id, tc):
        r = tc.endpoint_autoactivate(ep_id, if_expires_in=3600)
        if r['code'] == 'AutoActivationFailed':
            print('Endpoint({}) Not Active! Error! Source message: {}'.format(ep_id, r['message']))
            print("Endpoint requires manual activation, please open "
                  "the following URL in a browser to activate the "
                  "endpoint:")
            print("https://app.globus.org/file-manager?origin_id=" + ep_id)
            # For python 3.X, use input() instead
            input("Press ENTER after activating the endpoint:")
            r = tc.endpoint_autoactivate(ep_id, if_expires_in=3600)
        elif r['code'] == 'AutoActivated.CachedCredential':
            print('Endpoint({}) autoactivated using a cached credential'.format(ep_id))
        elif r['code'] == 'AutoActivated.GlobusOnlineCredential':
            print(('Endpoint({}) autoactivated using a built-in Globus credential.').format(ep_id))
        elif r['code'] == 'AlreadyActivated':
            print('Endpoint({}) already active until at least {}'.format(ep_id, 3600))
        else:
            print(r)


    import globus_sdk
    import webbrowser

    UUIDS = dict(
    LAPTOP = 'bae1b652-dbf1-11ea-85a2-0e1702b77d41',
    NERSC_DTN = '9d6d994a-6d04-11e5-ba46-22000b92c6ec',
    NERSC_COLLAB = 'df82346e-9a15-11ea-b3c4-0ae144191ee3',
    #PHASIS_DTN = '65ed6c76-76e3-11e6-8432-22000b97daec', # doesn't work
    )

    CLIENT_ID = '4fb1257f-31d8-4a0b-b91b-21d3198ea184'
    PHASIS_DTN_ROOT = '/cosmic-dtn/groups/cosmic/Data/'
    COLLAB_DTN_ROOT = '/global/cfs/cdirs/als/compute/test/'
    NERSC_DTN_ROOT = '/global/cfs/cdirs/als/test/'

    tokens = None
    try:
        # if we already have tokens, load and use them
        tokens = load_tokens_from_file(TOKEN_FILE)
    except:
        pass

    if not tokens:
        client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
        scopes = 'openid email profile urn:globus:auth:scope:transfer.api.globus.org:all'
        #scopes+= ' urn:globus:auth:scope:auth.globus.org:view_identity_set'
        #scopes+= ' urn:globus:auth:scope:auth.globus.org:view_identities'
        client.oauth2_start_flow(requested_scopes=scopes)

        authorize_url = client.oauth2_get_authorize_url()
        print('Please go to this URL and login: {0}'.format(authorize_url))
        webbrowser.open(authorize_url, new=1)

        auth_code = input(
            'Please enter the code you get after login here: ').strip()
        token_response = client.oauth2_exchange_code_for_tokens(auth_code)

        #globus_auth_data = token_response.by_resource_server['auth.globus.org']
        tokens = token_response.by_resource_server #['transfer.api.globus.org']

        try:
            save_tokens_to_file(TOKEN_FILE, tokens)
        except:
            pass

    # most specifically, you want these tokens as strings
    #AUTH_TOKEN = globus_auth_data['access_token']
    transfer_tokens = tokens["transfer.api.globus.org"]
    auth_client = globus_sdk.NativeAppAuthClient(client_id=CLIENT_ID)
    authorizer = globus_sdk.RefreshTokenAuthorizer(
        transfer_tokens["refresh_token"],
        auth_client,
        access_token=transfer_tokens["access_token"],
        expires_at=transfer_tokens["expires_at_seconds"],
        on_refresh=update_tokens_file_on_refresh,
    )

    #authorizer = globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
    tc = globus_sdk.TransferClient(authorizer=authorizer)

    for uuid in UUIDS.values():
        activate_endpoint(uuid, tc)
else:
    tc = None


P = Processor(pars)
P.PROCESSED_FRAMES_PER_NODE = 10

parallel.barrier()

keys=set()
prepared = False
initialized = False
scan_keys = []
ii = 0
while True:
    ii+=1
    time.sleep(0.1) # heart beat
    state = ''
    if parallel.master:
        S.process_requests()
        nkeys = set(S.objects.keys())
        diff = nkeys - keys
        if not diff:
            # nothing changed
            state = 'No change'
        else:
            keys = nkeys
            for key in diff:
                #print(ii, key)
                if "info" in key:
                    scan_keys.append(key)
            state = 'New data'

        # no scan_keys also means to continue
        if not scan_keys:
            print(ii, "No keys")
            time.sleep(0.5)
            continue

    #print(ii, parallel.rank)
    # ok now we have at least one scan, synchronize other ranks
    parallel.barrier()

    if not prepared:
        print(parallel.rank, "Preparing")
        # This part is blocking
        if parallel.master:
            sk = scan_keys[0]
            scan = S.objects[sk]
            scan = parallel.bcast(scan)
        else:
            scan = parallel.bcast(None)

        P.load_scan_definition(scan)

        if parallel.master:
            print(verbose.report(P.param))

        P.calculate_geometry()
        P.calculate_translation()
        # P.prepare(dark_dir_ram='/dev/shm/dark',exp_dir_ram='/dev/shm/exp')
        iadict = S.objects if parallel.master else None
        P.prepare(dealer=IAServerDealer, iadict=iadict)
        prepared = True

    if not initialized:
        initialized = P.process_init(start=0)
        #print(parallel.rank, "Initialized %s" % initialized)
        continue

    if not P.end_of_scan:
        #print(parallel.rank, "Enter loop ...")
        msg = P.process_loop()
        if msg == 'break':
            break
        elif msg == 'wait':
            continue
        else:
            fmain, fchunk, nchunk = msg
            if use_globus and parallel.master:
                tdata = globus_sdk.TransferData(tc,
                                            UUIDS['LAPTOP'],
                                            UUIDS['NERSC_DTN'],
                                            label="cosmic-stream-chunk%d" %nchunk, #os.path.split(fchunk)[1],
                                            sync_level="checksum",
                                            preserve_timestamp=True,
                                            notify_on_succeeded=True if nchunk==0 else False)
                main_dest_path = NERSC_DTN_ROOT+os.path.split(fmain)[1]
                tdata.add_item(fmain, main_dest_path)
                tdata.add_item(fchunk, NERSC_DTN_ROOT+os.path.split(fchunk)[1])
                transfer_result = tc.submit_transfer(tdata)
                if nchunk==0:
                    # Wait for first transmission
                    while not tc.task_wait(transfer_result['task_id'], timeout=10):
                        print("Waiting for chunk 0 transfer to complete")
                    task_id = api("compute/jobs/cori", {"job": api_job % main_dest_path, "isPath": "false"}, as_form=True)['task_id']
                    print("Job submitted for "+main_dest_path)
    else:
        P.print_benchmarks()
        P.write_cxi()
        if parallel.master:
            scan_keys.pop(0)
        prepared = False
        initialized = False
