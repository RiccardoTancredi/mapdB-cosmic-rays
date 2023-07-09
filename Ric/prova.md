Before starting to calculate the trajectories for each chamber for each event, we need to make sure to remove most of the noise (if not all) or the clearly wrong hits.
Let's start showing some examples of good events:
<p align = 'center'>
    <img src='./imgs/good_cell1.PNG' />
    <img src='./imgs/good_cell2.PNG' />
    <img src='./imgs/good_cell3.PNG' />
</p>
In these cases all the cells that gave a signal are only the ones that make up the trajectory.

But there are some case where other cells generated a signal:
<p align = 'center'>
    <img src='./imgs/wrong_cells_1.PNG' />
    <img src='./imgs/wrong_cells_2.PNG' />
</p>

We would like to filter out these easy-to-spot cases. To solve this we developed an algorithm to group the cells by proximity and then discard the groups with too few or too many hits [magari specificare quanti]. Here we show the results for the case seen before:
<p align = 'center'>
    <img src='./imgs/wrong_cells_group_1.PNG' />
    <img src='./imgs/wrong_cells_group_2.PNG' />
</p>

The next step is to calculate the right trajectory by solving the left-right ambiguity. We do that by selecting every possible left right combination for cell, so if we have 4 cells, we have 16 possible combination of left an right. For every combination we calculate the linear regression of the points and then we sum the squared distance from the points to the line. We choose the line with the lowest sum, but we also keep the second-best line.

<p align = 'center'>
    <img src='./imgs/good_lines1.PNG' />
    <img src='./imgs/good_lines2.PNG' />
</p>

[global tracks]

Then we need to calculate the global track. To do that we use the set of points of the best and second-best lines for each chamber. We try every possible combination of the best and second best set of points to choose the best global line in the same way as before. We do that because, for example, the set of points of the best line for a chamber could not be the best set of points for the global line, but the second-best set could fit better. In this case there are at maximum of 3 chambers so there a maximum of 8 combinations of best and second-best.

<p align = 'center'>
    <img src='./imgs/global_1.PNG' />
    <img src='./imgs/global_2.PNG' />
    <img src='./imgs/global_3.PNG' />
</p>