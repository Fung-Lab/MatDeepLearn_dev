import globus_sdk
from globus_sdk.scopes import TransferScopes

CLIENT_ID = "4f6c2c11-6031-4e30-831b-2a3805e1213a"
auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)

# requested_scopes specifies a list of scopes to request
# instead of the defaults, only request access to the Transfer API
auth_client.oauth2_start_flow(requested_scopes=TransferScopes.all)
authorize_url = auth_client.oauth2_get_authorize_url()
print(f"Please go to this URL and login:\n\n{authorize_url}\n")

auth_code = input("Please enter the code here: ").strip()
tokens = auth_client.oauth2_exchange_code_for_tokens(auth_code)
transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]

# construct an AccessTokenAuthorizer and use it to construct the
# TransferClient
transfer_client = globus_sdk.TransferClient(
    authorizer=globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
)

# Globus Tutorial Endpoint 1
source_endpoint_id = "9d6d994a-6d04-11e5-ba46-22000b92c6ec"
# Globus Tutorial Endpoint 2
dest_endpoint_id = "9f127ec0-edff-11ed-ba42-09d6a6f08166"

# create a Transfer task consisting of one or more items
task_data = globus_sdk.TransferData(
    source_endpoint=source_endpoint_id, destination_endpoint=dest_endpoint_id
)
task_data.add_item(
    "/global/cfs/projectdirs/m3641/Shared/Materials_datasets/hMOF/raw_5k/ocp_3",  # source
    "/nethome/sbaskaran31/projects/Sidharth/hMOF/raw_5k/",  # dest
    recursive=True,
)

# submit, getting back the task ID
task_doc = transfer_client.submit_transfer(task_data)
task_id = task_doc["task_id"]
print(f"submitted transfer, task_id={task_id}")