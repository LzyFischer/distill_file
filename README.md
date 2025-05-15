1. 直接运行cot_pipeline.py就可以了。但是需要注意以下几个地方：
   1. 设置max_output_token为1000会不会有问题，这个是follow reverse thinking的设定。
   2. 开头的时候需要检查调用api function的路径是什么
   3. args的模型看是pro还是flash
2. 其他地方都test过没有问题了，但如果还有什么bug的话可以把bug的问题发给我，我可以再debug一下。
   1. 可以先用tmp dataset调试一下。   