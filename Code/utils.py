import matplotlib.pyplot as plt

def append_history(history, h):
     '''
	This function appends the statistics over epochs
     '''
     try:
       history.history['loss'] = history.history['loss'] + h.history['loss']
       history.history['val_loss'] = history.history['val_loss'] + h.history['val_loss']
       history.history['acc'] = history.history['acc'] + h.history['acc']
       history.history['val_acc'] = history.history['val_acc'] + h.history['val_acc']
     except:
       history = h
                
     return history
            

def unfreeze_layer_onwards(model, layer_name):
    '''
        This layer unfreezes all layers beyond layer_name
    '''
    trainable = False
    for layer in model.layers:
        try:
            if layer.name == layer_name:
                trainable = True
            layer.trainable = trainable
        except:
            continue
    
    return model
            

def plot_performance(history):
    '''
	This function plots the train & test accuracy, loss plots
    '''
        
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy v/s Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left') 

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss v/s Epochs')
    plt.ylabel('M.S.E Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left') 

    plt.tight_layout()
    plt.show()

    
