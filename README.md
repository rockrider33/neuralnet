# neuralnet
Repo for neural net biz.

<h3><b>Setup:</b></h3>
Currently using virtualbox VM with ububtu 16.04 and 3GB memory allotted...it's working fine with mnist database. Had seen that VM is struggling with RNN and CNN for the same dataset. 


<h3><b>Accuracy</b></h3>

<h4><b>Vanilla NN:</b></h4>
<ul>
<li>3 hidden each has 500 neurons and epoch of 10: 0.95190001</li>
<li>3 hidden each has 500 neurons and epoch of 20: 0.96219999</li>
</ul>
<h4><b>RNN:</b></h4>
RNN size:128 , chunk/sequence/timestep size:28 (which is image width), epoch:10 : 0.98369998
