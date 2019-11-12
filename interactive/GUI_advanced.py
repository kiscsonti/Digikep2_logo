#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wx
import os
wildcard = "Image Files (*.png;*.jpg)|*.png;;*.jpg|All Files (*.*)|*.*"

import torch
from utils import *

class Example(wx.Frame):

    def __init__(self, *args, **kw):
        super(Example, self).__init__(*args, **kw)

        PATH = "/home/petigep/college/orak/digikep2/saved_states/75percent_gold/best_model_1234.mdl"
        self.model = torch.load(PATH)

        self.seed = 1234
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.id2label = {0: 'adidas',
                         1: 'apple',
                         2: 'cocacola',
                         3: 'disney',
                         4: 'nike',
                         5: 'nologo',
                         6: 'puma'}

        self.InitUI()

    def scale_bitmap(self, image:wx.Image):
        w = image.GetWidth()
        h = image.GetHeight()
        if w > h:
            ratio = 512/w
        else:
            ratio = 512/h
        image = image.Scale(int(w*ratio), int(h*ratio), wx.IMAGE_QUALITY_HIGH)
        result = wx.BitmapFromImage(image)
        return result

    def InitUI(self):
        self.SetSize(wx.Size(1000, 800))
        self.vbox = wx.BoxSizer(wx.VERTICAL)

        self.panel = wx.Panel(self)

        file_drop_target = FileDrop(self)
        self.panel.SetDropTarget(file_drop_target)

        self.currentDirectory = os.getcwd()

        self.predLabel = wx.StaticText(self.panel, -1, style=wx.ALIGN_CENTER)
        font1 = wx.Font(26, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.predLabel.SetFont(font1)
        self.predLabel.SetLabelText("Itt lesz a modell válasza")


        self.moreInfoLabel = wx.StaticText(self.panel, -1, style=wx.ALIGN_CENTER)
        # font2 = wx.Font(20, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        font2 = wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.moreInfoLabel.SetFont(font2)
        self.moreInfoLabel.SetLabelText("Itt lesz a modell válasza minden cimkére")

        self.browseButton = wx.Button(self.panel, wx.ID_ANY, 'Browse', size=(90, 30))
        self.browseButton.Bind(wx.EVT_BUTTON, self.onOpenFile)
        self.textField = wx.TextCtrl(self.panel, wx.ID_ANY, "", size=(800, 35))
        self.Bind(wx.EVT_BUTTON, self.NewItem, id=self.browseButton.GetId())
        # self.Bind(wx.EVT_LISTBOX_DCLICK, self.OnRename)
        font3 = wx.Font(16, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        self.textField.SetFont(font3)


        imageFile = r"/home/petigep/college/orak/digikep2/Digikep2_logo/InteractWIthModel/thumbnail.png"
        # imageFile = r"E:\egyetem\mester\1.felev\demonstrator\alga1\memes\meme_for_pluszpont.png"
        self.load_image(imageFile, init=True)


        self.vbox.Add(self.picture, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)

        # vbox.Add((-1, 20))
        self.vbox.Add(self.predLabel, 0, wx.CENTER)
        self.vbox.Add((-1, 10))
        self.vbox.Add(self.moreInfoLabel, 0, wx.CENTER)
        self.vbox.Add((-1, 10))
        self.vbox.Add(self.browseButton, 0, wx.CENTER)
        self.vbox.Add((-1, 10))
        self.vbox.Add(self.textField, 0, wx.CENTER)
        self.vbox.Add((-1, 20))
        self.panel.SetSizer(self.vbox)

        self.SetTitle('Interact With Model')
        self.Centre()

    def onOpenFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory,
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            print("You chose the following file(s):")
            for path in paths:
                self.load_image(path)
                print(path)

        dlg.Destroy()

    def load_image(self, path, init=False):

        img1 = wx.Image(path)
        img1 = self.scale_bitmap(img1)
        self.textField.SetLabelText(path)
        if init:
            self.picture = wx.StaticBitmap(self.panel, -1, wx.BitmapFromImage(img1))
        else:
            self.picture.Hide()
            self.picture.SetBitmap(wx.BitmapFromImage(img1))
            self.picture.Show()
        self.vbox.Layout()

        self.predict(path)

        self.Refresh()
        self.Update()

    def predict(self, path):

        img_tensor = get_image_tensor(path)
        # print(img_tensor.size())

        pred = self.model(img_tensor)
        print(pred)
        label = np.argmax(pred.tolist())

        self.predLabel.SetLabelText(str(label) + " : " + self.id2label[label])

        info = ", ".join([str(self.id2label[index]) + ": " + "{:.4f}".format(float(np.power(np.e, item))) for index, item in enumerate(pred.tolist()[0])])
        self.moreInfoLabel.SetLabelText(info)

        pass


    def NewItem(self, event):

        text = wx.GetTextFromUser('Enter a new item', 'Insert dialog')
        if text != '':
            self.listbox.Append(text)


class FileDrop(wx.FileDropTarget):

    def __init__(self, window):

        wx.FileDropTarget.__init__(self)
        self.window = window

    def OnDropFiles(self, x, y, filenames):

        for name in filenames:
            print(name)
            self.window.load_image(name)


        return True






def main():

    app = wx.App()
    ex = Example(None)
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
