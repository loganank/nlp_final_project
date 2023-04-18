from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch


class BERTClass(torch.nn.Module):
    def __init__(self, NUM_OUT):
        super(BERTClass, self).__init__()

        self.l1 = BertModel.from_pretrained("bert-base-uncased")  # roberta-base # bert-base-uncased
        self.classifier = torch.nn.Linear(768, NUM_OUT)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        output = self.classifier(pooler)
        output = self.softmax(output)
        return output


app = Flask(__name__)
saved_model_path = 'F:\\GitHub\\nlp_final_project\\flask-server\\saved_model'
model = BERTClass(3)
model = torch.load(saved_model_path, map_location=torch.device('cpu'))

labels = {
    0: 'suicidal',
    1: 'depressed',
    2: 'not suicidal or depressed'
}


# users API Route
@app.route('/api')
def users():
    return {'users': ['userOne', 'userTwo', 'userThree', 'userFour']}


@app.route('/evaluateText', methods=['POST'])
def evaluate_text():
    # where the text is fed to model and output is returned
    input_text = request.get_json()
    print(input_text)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokenized_input = tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokenized_input)

    # Get the predicted label
    predicted_label = int(torch.argmax(output))

    # Print the predicted label
    print(labels[predicted_label])

    return jsonify(
        status='ok',
        classifier=labels[predicted_label]
    )


if __name__ == '__main__':
    app.run(debug=True)