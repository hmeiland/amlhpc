@description('Specifies the name of the deployment.')
param name string

var location = resourceGroup().location

var resourcePostfix = '${uniqueString(subscription().subscriptionId, resourceGroup().name)}z'

var tenantId = subscription().tenantId
var storageAccountName = substring('st${name}${resourcePostfix}',0,20)
var keyVaultName = substring('kv${name}${resourcePostfix}',0,20)
var applicationInsightsName = substring('ai${name}${resourcePostfix}',0,20)
var containerRegistryName = substring('cr${name}${resourcePostfix}',0,20)
var workspaceName = substring('ml${name}${resourcePostfix}',0,20)
var virtualNetworkName = substring('vn${name}${resourcePostfix}',0,20)
var storageAccountId = storageAccount.id
var keyVaultId = vault.id
var applicationInsightId = applicationInsight.id
var containerRegistryId = registry.id
var subnetClusterId = subnetCluster.id

resource storageAccount 'Microsoft.Storage/storageAccounts@2022-05-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_RAGRS'
  }
  kind: 'StorageV2'
  properties: {
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource vault 'Microsoft.KeyVault/vaults@2022-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    accessPolicies: []
    enableSoftDelete: true
  }
}

resource applicationInsight 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}

resource registry 'Microsoft.ContainerRegistry/registries@2022-02-01-preview' = {
  sku: {
    name: 'Standard'
  }
  name: containerRegistryName
  location: location
  properties: {
    adminUserEnabled: false
  }
}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2023-06-01-preview' = {
  identity: {
    type: 'SystemAssigned'
  }
  name: workspaceName
  location: location
  properties: {
    friendlyName: workspaceName
    storageAccount: storageAccountId
    keyVault: keyVaultId
    applicationInsights: applicationInsightId
    containerRegistry: containerRegistryId
  }
}

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2021-05-01' = {
  name: virtualNetworkName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/16'
      ]
    }
  }
}

resource subnetCluster 'Microsoft.Network/virtualNetworks/subnets@2021-08-01' = {
  parent: virtualNetwork
  name: 'cluster'
  properties: {
      addressPrefix: '10.0.1.0/24'
  }
}

resource subnetanf 'Microsoft.Network/virtualNetworks/subnets@2021-08-01' = {
  parent: virtualNetwork
  name: 'anf'
  properties: {
      addressPrefix: '10.0.2.0/24'
  }
}

resource amlLoginVM 'Microsoft.MachineLearningServices/workspaces/computes@2023-06-01-preview' = {
  parent: workspace
  name: 'login-${resourcePostfix}' 
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    computeType: 'ComputeInstance'
    computeLocation: location
    description: 'Login vm'
    disableLocalAuth: true
    properties: {
      applicationSharingPolicy: 'Personal'
      
      computeInstanceAuthorizationType: 'personal'
      sshSettings: {
        sshPublicAccess: 'Disabled'
      }
      subnet: {
        id: subnetClusterId
      }
      customServices: [{
        environmentVariables: {
          test_variable: {
            type: 'local'
            value: '12345'
          }
        }
      }] 
      vmSize: 'Standard_F2s_v2'
      setupScripts: {
        scripts: {
          creationScript: {
	    scriptSource : 'inline'
	    scriptData: base64('''
pip install amlhpc 
#echo "export SUBSCRIPTION=${subscription().subscriptionId}" > /etc/profile.d/amlhpc.sh
echo "export SUBSCRIPTION=$test_variable" > /etc/profile.d/amlhpc.sh
''')
          }
          startupScript: {
	    scriptSource : 'inline'
	    scriptData: base64('echo `date` > /startup.txt')
          }
        }
      }
    }
  }
}

resource smallcluster 'Microsoft.MachineLearningServices/workspaces/computes@2023-06-01-preview' = {
  parent: workspace
  name: 'f4s'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    computeType: 'AmlCompute'
    computeLocation: location
    disableLocalAuth: true
    properties: {
      vmPriority: 'Dedicated'
      vmSize: 'Standard_F4s_v2' 
      //enableNodePublicIp: amlComputePublicIp
      isolatedNetwork: false
      osType: 'Linux'
      remoteLoginPortPublicAccess: 'Disabled'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 5
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
      subnet: {
        id: subnetClusterId
      }
    }
  }
}

resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: resourceGroup()
  name: guid(amlLoginVM.name, resourcePostfix)
  properties: {
    roleDefinitionId: resourceId('microsoft.authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c')
    principalId: amlLoginVM.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource ml_cust_env 'Microsoft.MachineLearningServices/workspaces/environments/versions@2023-06-01-preview' = {  
  name: '${workspace.name}/amlhpc-ubuntu2004/1'  
  properties: {  
    osType: 'Linux'
    image: 'docker.io/hmeiland/amlhpc-ubuntu2004'
    autoRebuild: 'OnBaseImageUpdate'
  }  
}  
