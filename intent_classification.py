
'''
Name: Tuhin Kundu
Graduate student, Computer Science, University of Illinois at Chicago

Rough idea of the pipeline for intent classification.
Code tested on local system with RTX 2060 GPU
'''


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
from random import shuffle
import time
import datetime
import json
import os
import sklearn

class pipeline:

    def __init__(self, num_classes, tokenizer, model_version = 'bert-base-uncased'):

        self.num_labels = num_classes

        self.tokenizer = tokenizer

        self.model = BertForSequenceClassification.from_pretrained(model_version, num_labels=self.num_labels,
                                                              output_attentions=False, output_hidden_states=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.softmax = torch.nn.Softmax()

    def get_optimizer(self, lr=2e-5, eps=1e-8):
        optimizer = AdamW(self.model.parameters(), lr=lr, eps=eps)
        return optimizer

    def get_scheduler(self, optimizer, total_steps, warmup_steps=100):
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        return scheduler

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def loop_over(self, data, classes):
        select_data, select_labels = [], []
        shuffle(data)
        for row in data:
            if row[1] in classes:
                select_data.append(row[0])
                select_labels.append(row[1])
        return select_data, select_labels

    def prepare_splits(self, data):

        s = set()
        for row in data['train']:
            s.add(row[-1])

        classes = list(s)[:self.num_labels]

        train_data, train_labels = self.loop_over(data['train'], classes)
        val_data, val_labels = self.loop_over(data['val'], classes)
        test_data, test_labels = self.loop_over(data['test'], classes)

        return classes, [train_data, train_labels], [val_data, val_labels], [test_data, test_labels]

    def labels2idx_func(self, labels, labels2idx):

        idx_labels = []
        for label in labels:
            idx_labels.append(labels2idx[label])
        return idx_labels

    def encode_sentences(self, data, max_length=64):
        input_ids = []
        attention_masks = []
        for sent in data:
            encoded = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=max_length, pad_to_max_length=True,
                                            return_attention_mask=True, return_tensors='pt')
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks


    def prepare_inputs(self, data, batch_size=64):

        label_set, train, val, test = self.prepare_splits(data)

        labels2idx = {label: i for i, label in enumerate(label_set)}

        train_data, train_labels = train[0], self.labels2idx_func(train[1], labels2idx)
        val_data, val_labels = val[0], self.labels2idx_func(val[1], labels2idx)
        test_data, test_labels = test[0], self.labels2idx_func(test[1], labels2idx)

        train_input_ids, train_mask = self.encode_sentences(train_data)
        val_input_ids, val_mask = self.encode_sentences(val_data)
        test_input_ids, test_mask = self.encode_sentences(test_data)

        train_dataset = TensorDataset(train_input_ids, train_mask, torch.tensor(train_labels))
        val_dataset = TensorDataset(val_input_ids, val_mask, torch.tensor(val_labels))
        test_dataset = TensorDataset(test_input_ids, test_mask, torch.tensor(test_labels))

        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
        validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

        return labels2idx, train_dataloader, validation_dataloader, test_dataloader


    def load_data(self, filename):

        with open(filename, 'r') as f:
            data = json.load(f)

        return data

    def train(self, train_data, val_data, epochs, optimizer, scheduler, seed=42):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.device == 'gpu':
            torch.cuda.manual_seed_all(seed)

        training_stats = []
        total_t0 = time.time()
        for epoch_i in range(0, epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            t0 = time.time()
            total_train_loss = 0
            self.model.train()
            for step, batch in enumerate(train_data):

                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data), elapsed))

                # loading data to whichever device available
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()

                # Sending data to the model and storing loss and logits generated by the model.
                loss, logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                total_train_loss += loss.item()

                loss.backward()  # backpropagating after loss calculation

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               1.0)  # gradient clipping in case of gradient explosion

                # running the optimizer and the scheduler for the learning rate
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_data)

            training_time = self.format_time(time.time() - t0)
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
            print("Running Validation...")

            t0 = time.time()

            # Validation
            self.model.eval()
            total_eval_accuracy = 0
            total_eval_loss = 0

            for batch in val_data:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad():  # no need of loss propagation

                    (loss, logits) = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                                           labels=b_labels)

                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            avg_val_accuracy = total_eval_accuracy / len(val_data)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
            avg_val_loss = total_eval_loss / len(val_data)
            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time() - total_t0)))

    def testing(self, test_data):

        print('Predicting labels')

        self.model.eval()

        predictions, true_labels = [], []

        for batch in test_data:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.append(logits)
            true_labels.append(label_ids)
        return predictions, true_labels

    def process_logits_and_labels(self, logits, labels):
        labels_flat = []
        pred_flat = []
        for i, batch in enumerate(logits):
            for j, logits in enumerate(batch):
                pred_flat.append(np.argmax(self.softmax(torch.tensor(logits))))
                labels_flat.append(labels[i][j])
        return labels_flat, pred_flat

    def get_prediciton_accuracy_f1_score(self, predictions, labels):
        cnt = 0
        for i, val in enumerate(predictions):
            if val == labels[i]:
                cnt += 1
        print('Accuracy of the model on the test set')
        accuracy = cnt / len(labels)
        print(accuracy)

        print('Macro averaged F1 score')
        f1 = sklearn.metrics.f1_score(labels, predictions, average='macro')
        print(f1)

        return accuracy, f1

    def save_trained_model(self, outpath):

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if not os.path.isfile(outpath):
            os.mkdir(outpath)
        model_to_save.save_pretrained(outpath)
        self.tokenizer.save_pretrained(outpath)


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    pipe = pipeline(num_classes=20, tokenizer=tokenizer)
    pathname = '../data_full.json'
    data = pipe.load_data(filename=pathname)

    labels2idx, train_dataloader, validation_dataloader, test_dataloader = pipe.prepare_inputs(data=data)

    epochs = 5
    total_steps = len(train_dataloader) * epochs
    optimizer = pipe.get_optimizer()
    scheduler = pipe.get_scheduler(optimizer=optimizer, total_steps=total_steps)

    pipe.train(train_data=train_dataloader, val_data=validation_dataloader,
               epochs=epochs, optimizer=optimizer, scheduler=scheduler, seed=41)

    logits, ground_truth = pipe.testing(test_data=test_dataloader)
    predictions, labels = pipe.process_logits_and_labels(logits, ground_truth)
    accuracy, f1_score = pipe.get_prediciton_accuracy_f1_score(predictions=predictions, labels=labels)
    pipe.save_trained_model('output_dir')












