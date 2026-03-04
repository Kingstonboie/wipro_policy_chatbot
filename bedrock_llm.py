# bedrock_llm.py
import boto3
import json
from config import AWS_REGION, BEDROCK_MODEL_ID

class BedrockLLM:
    """Simple wrapper for AWS Bedrock that mimics Ollama's interface"""
    
    def __init__(self, model_id=BEDROCK_MODEL_ID, region=AWS_REGION):
        self.model_id = model_id
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region
        )
        print(f"✅ Initialized Bedrock client with model: {model_id}")
    
    def invoke(self, prompt):
        """Invoke Bedrock model with prompt (matches Ollama's invoke method)"""
        
        # Format request based on model type
        if "anthropic.claude" in self.model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}]
            })
        elif "amazon.titan" in self.model_id:
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 1000,
                    "temperature": 0.7,
                    "topP": 0.95
                }
            })
        else:
            # Generic format for other models
            body = json.dumps({"prompt": prompt, "max_tokens": 1000})
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            
            result = json.loads(response['body'].read())
            
            # Parse response based on model type
            if "anthropic.claude" in self.model_id:
                return result['content'][0]['text']
            elif "amazon.titan" in self.model_id:
                return result['results'][0]['outputText']
            else:
                return str(result)
                
        except Exception as e:
            print(f"❌ Bedrock error: {e}")
            return f"Error: {str(e)}"