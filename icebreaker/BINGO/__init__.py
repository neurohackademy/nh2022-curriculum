# -*- coding: utf-8 -*-
################################################################################
# BINGO!
#
# This library makes BINGO sheets for the NeuroHackademy 2022.
#
# @author Noah C. Benson <nben@uw.edu>

# Dependencies #################################################################
import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

# Configuration ################################################################

# Make sure we can use the Helvetica Neue font (if it exists) and that we use
# helvetica or arial if not.
mpl_font_config = {'family':'sans-serif',
                   'sans-serif':['HelveticaNeue', 'Helvetica', 'Arial'],
                   'size': 10,
                   'weight': 'light'}
mpl.rc('font', **mpl_font_config)

# Colors for the squares.
physical_color = mpl.colors.to_rgba("#CCE3F9")
virtual_color = (1,1,1,1)
either_color = (0.8, 0.8, 0.8, 1)

# Questions
physical_qs = [
        "... who is wearing a button-up shirt.",
        "... who has traveled at least 1000 miles to Seattle.",
        "... who lives in Seattle.",
        "... who has never attended an in-person conference before."]
virtual_qs = [
        "... who is Zooming from outside.",
        "... who has at least 4 plants in the background.",
        "... who is currently on an island.",
        "... who has a cat in their background."]
either_qs = [
        "... who is a morning person.",
        "... who speaks 2 languages not including English.",
        "... who would rather be able to fly than to be invisible.",
        "... who would rather be telepathic than telekinetic.",
        "... who would rather have X-ray vision than the ability to see the future.",
        "... who is a member of a band or a choir.",
        "... who has had their appendix removed.",
        "... whose graduate work is/was not related to neuroscience.",
        "... who goes primarily by a nickname.",
        "... who is vegetarian or vegan.",
        "... who works with PET data.",
        "... who works with MEG or EEG data.",
        "... who has never broken a bone.",
        "... who has read but not watched Game of Thrones.",
        "... who has a vegetable or herb garden.",
        "... who has a flower garden.",
        "... who belongs to a book club.",
        "... who prefers chocolate over vanilla dessert.",
        "... who is an oldest or only child.",
        "... who is a youngest child.",
        "... who does crosswords regularly.",
        "... who has a painting or knitting hobby.",
        "... who is far-sighted (hyperopic).",
        "... who is not active on Twitter.",
        "... who prefers biking over hiking.",
        "... who maintains at least one Python library on PyPI.",
        "... who prefers video games over movies or television.",
        "... who has a non-natural hair color (e.g., green, purple)."]

# The BINGO grids.
physical_tag = -1
virtual_tag = 1
either_tag = 0
(p,v,e) = (physical_tag, virtual_tag, either_tag)
virtual_grid = [[e, v, p, v, p, e],
                                [v, v, p, e, p, p],
                                [p, e, p, v, p, v],
                                [v, p, v, p, e, p],
                                [p, p, e, p, v, v],
                                [e, p, v, p, v, e]]
virtual_grid = np.array(virtual_grid)
physical_grid = -virtual_grid
virtual_img = np.zeros(virtual_grid.shape + (4,), dtype='float')
physical_img = np.zeros(virtual_grid.shape + (4,), dtype='float')
for (tag,clr) in zip([virtual_tag,   physical_tag,   either_tag],
                     [virtual_color, physical_color, either_color]):
    virtual_img[virtual_grid == tag, :] = clr
    physical_img[physical_grid == tag, :] = clr

# Aliases for conventient automation:
tags = {'physical': physical_tag,
        'virtual': virtual_tag,
        'either': either_tag}
grids = {'physical': physical_grid,
         'virtual': virtual_grid}
backgrounds = {'physical': physical_img,
               'virtual': virtual_img}
questions = {'physical': physical_qs,
             'virtual': virtual_qs,
             'either': either_qs}
colors = {'physical': physical_color,
          'virtual': virtual_color,
          'either': either_color}

# Rooms:
rooms_physical = ['AA', 'A103', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'C', 'L']
probs_physical = np.array([1]*8 + [3, 3])
probs_physical = probs_physical / np.sum(probs_physical)
rooms_virtual = ['AA', 'A103', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'Z1', 'Z2']
probs_virtual = np.array([1]*8 + [3, 3])
probs_virtual = probs_virtual / np.sum(probs_virtual)
nrooms = len(rooms_physical)
def roomdraw(style='virtual', nrounds=6):
    from numpy.random import choice
    if style == 'virtual':
        return choice(rooms_virtual, nrounds, replace=True, p=probs_virtual)
    else:
        return choice(rooms_physical, nrounds, replace=True, p=probs_physical)

# BINGO Sheets! ################################################################
def draw(style='virtual', header_width=5.5, header_height=2.75, nrounds=6):
    from textwrap import wrap
    fig = plt.figure(figsize=(8.5,11), dpi=72*4, facecolor='white')
    # The background patch.
    ax_bg = fig.add_axes([0,0,1,1])
    ax_bg.add_patch(plt.Rectangle([0,0],1,1, edgecolor=None, facecolor='white'))
    ax_bg.axis('off')
    # The header image.
    x0 = (8.5 - header_width)/2
    x1 = (8.5 + header_width)/2
    y0 = 10.25 - header_height
    y1 = 10.25
    ax_header = fig.add_axes([x0/8.5, y0/11, (x1-x0)/8.5, (y1-y0)/11])
    ax_header.set_xlim([x0,x1])
    ax_header.set_ylim([y0,y1])
    ax_header.text((x0+x1)/2, y1,
                   'NeuroHackademy BINGO!',
                   va='top', ha='center',
                   fontsize=42)
    ax_header.text((x0+x1)/2, y0 + 0.775*(y1 - y0),
                   ('Be the first to find 6 participants in a\n'
                    'row, column, or diagonal to win!'),
                   va='top', ha='center',
                   fontsize=20)
    rooms = roomdraw(style, nrounds=nrounds)
    ax_header.text((x0+x1)/2, y0 + 0.5*(y1 - y0),
                   ('Rooms: ' + ', '.join(rooms)),
                   va='top', ha='center',
                   fontsize=16)
    s = (x1-x0)/6
    (rx0,ry0) = (x0+(x1-x0)/12, y0+0.1)
    r1 = plt.Rectangle([rx0, ry0], s, s,
                       facecolor=colors['physical'],
                       edgecolor='k', lw=1)
    ax_header.add_patch(r1)
    ax_header.text(rx0+s/2, ry0 + s*0.75, 'Find an', ha='center', va='center')
    ax_header.text(rx0+s/2, ry0 + s*0.5, 'in-person', ha='center', va='center', weight='bold')
    ax_header.text(rx0+s/2, ry0 + s*0.25, 'participant ...', ha='center', va='center')
    (rx0,ry0) = (x0+(x1-x0)*5/12, y0+0.1)
    r2 = plt.Rectangle([rx0, ry0], s, s,
                       facecolor=colors['either'],
                       edgecolor='k', lw=1)
    ax_header.add_patch(r2)
    ax_header.text(rx0+s/2, ry0 + s*0.75, 'Find', ha='center', va='center')
    ax_header.text(rx0+s/2, ry0 + s*0.5, 'any', ha='center', va='center', weight='bold')
    ax_header.text(rx0+s/2, ry0 + s*0.25, 'participant ...', ha='center', va='center')
    (rx0,ry0) = (x0+(x1-x0)*9/12, y0+0.1)
    r3 = plt.Rectangle([rx0, ry0], s, s,
                       facecolor=colors['virtual'],
                       edgecolor='k', lw=1)
    ax_header.add_patch(r3)
    ax_header.text(rx0+s/2, ry0 + s*0.75, 'Find an', ha='center', va='center')
    ax_header.text(rx0+s/2, ry0 + s*0.5, 'online', ha='center', va='center', weight='bold')
    ax_header.text(rx0+s/2, ry0 + s*0.25, 'participant ...', ha='center', va='center')
    ax_header.axis('off')
    # The squares.
    x0 = 1
    x1 = 8.5 - 1
    ymid = y0 / 2
    y0 = 0.75
    y1 = y0 + (x1-x0)
    ax_squares = fig.add_axes([x0/8.5, y0/11, (x1-x0)/8.5, (y1-y0)/11])
    bg = backgrounds[style]
    ax_squares.imshow(bg, extent=((0,) + bg.shape[:2] + (0,)))
    # Square ticks/frames.
    ax_squares.grid(which='major', color='k', linestyle='-', linewidth=1)
    ax_squares.set_frame_on(True)
    ax_squares.set_xticklabels([])
    ax_squares.set_yticklabels([])
    ax_squares.tick_params(color='w')
    # The text in the squares.
    # First pick the locations of the physical- and virtual-only squares.
    grid = grids[style]
    (pii,pjj) = np.where(grid == physical_tag)
    (vii,vjj) = np.where(grid == virtual_tag)
    ponly = np.random.choice(np.arange(len(pii)), len(physical_qs), replace=False)
    vonly = np.random.choice(np.arange(len(vii)), len(virtual_qs), replace=False)
    qs = np.full(grid.shape, '', dtype=object)
    qs[(pii[ponly],pjj[ponly])] = physical_qs
    qs[(vii[vonly],vjj[vonly])] = virtual_qs
    # Now, we fill the rest in with other questions.
    eord = np.random.choice(
        np.arange(len(either_qs)),
        grid.shape[0]*grid.shape[1] - len(physical_qs) - len(virtual_qs),
        replace=False)
    qs[qs == ''] = np.array(either_qs)[eord]
    # Okay, that gives us a grid of questions, now print them all.
    sq_x0s = np.arange(-0.5, grid.shape[1] - 0.5)
    sq_y0s = np.arange(-0.5, grid.shape[0] - 0.5)
    sq_x1s = np.arange(0.5, grid.shape[1] + 0.5)
    sq_y1s = np.arange(0.5, grid.shape[0] + 0.5)
    for (y0,y1,qrow) in zip(sq_y0s, sq_y1s, qs):
        for (x0,x1,q) in zip(sq_x0s, sq_x1s, qrow):
            ax_squares.text(x1, y1-0.475, '\n'.join(wrap(q, 14)),
                            va='top', ha='center')
    return fig
