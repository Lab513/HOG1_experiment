import os
op = os.path
opd, opb, opj, opa = op.dirname, op.basename, op.join, op.abspath
import btrack
from btrack.dataio import localizations_to_objects
from btrack.constants import BayesianUpdates
from btrack.render import plot_tracks


class APPLY_BTRACK(object):
    '''

    '''
    def find_tracks_btrack(self):
        '''
        '''
        objects = localizations_to_objects(self.pos_per_frm)
        with btrack.BayesianTracker() as tracker:
            tracker.configure_from_file(opj('settings','cell_config.json'))   # configure the tracker and change the update method
            tracker.update_method = BayesianUpdates.APPROXIMATE
            tracker.max_search_radius = 50
            tracker.append(objects)                                             # append the objects to be tracked
            tracker.volume=((0, 1200), (0, 1600), (-1e5, 1e5))                  # set the volume (Z axis volume is set very large for 2D data)
            tracker.track_interactive(step_size=100)                            # track them (in interactive mode)
            tracker.optimize()                                                  # generate hypotheses and run the global optimizer
            self.tracks = tracker.tracks                                        # get the tracks as a python list

    def debug_btrack_reformat(self,tt,i,t):
        '''
        '''
        print(f'tt {tt}')
        print(f'i {i}')
        print(f't.ID {t.ID}')

    def from_tracks_to_contours_format(self, debug=[]):
        '''
        Passing from btrack tracks format to pkl format
        '''
        print('#### Adapting btrack format ...')
        dic_cells = {}
        nb_frames = max(self.pos_per_frm['t'])+1
        nb_cells_max = 2*len(self.tracks)
        print(f'nb_frames {nb_frames}')
        print(f'nb_cells_max {nb_cells_max}')
        for fr in range(nb_frames):
            dic_cells[fr] = [None]*(nb_cells_max+1)
        for t in self.tracks:
            if 0 in debug: print(f"len(t['Contours'])  {len(t['Contours'])}")
            for i,tt in enumerate(t.t):
                if 0 in debug: self.debug_btrack_reformat(tt,i,t)
                dic_cells[tt][t.ID] = t['Contours'][i]

        return dic_cells
