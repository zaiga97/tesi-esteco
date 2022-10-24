import numpy as np
import pandas as pd
import math
from Data import RoadGeometry as rg

pd.options.mode.chained_assignment = None

CLOSE_TIME = 1
CLOSE_DIST = 2
MIN_POINTS = 10
MIN_DIST = 5
FREQ_SAMPLE = '250L'
STEP_PER_SECOND = 4
DEFAULT_UPPER = [20, 2, 0, 0]
DEFAULT_LOWER = [-20, -2, 0, 0]


# noinspection PyShadowingNames
def rotate_coord(df: pd.DataFrame, angle: float = 0):
    sin = math.sin(math.radians(angle))
    cos = math.cos(math.radians(angle))

    df['x1'] = df.x * cos - df.y * sin
    df['y1'] = df.x * sin + df.y * cos
    df = df.drop(labels=['x', 'y'], axis=1)
    df = df.rename(columns={'x1': 'x', 'y1': 'y'})
    return df


# noinspection PyShadowingNames
def translate_coord(df: pd.DataFrame, dx: float = 0, dy: float = 0):
    df.x = df.x + dx
    df.y = df.y + dy
    return df


def clean_data(original_df: pd.DataFrame):
    traj_df = original_df
    print('Changing coordinate system...')
    traj_df = rotate_coord(traj_df, angle=rg.ROTATION)
    traj_df = translate_coord(traj_df, dx=rg.X_TRANS, dy=rg.Y_TRANS)

    print('Calculating trajectories extrema...\n')
    # get the trajs extrema
    grouped_by_id = traj_df.groupby('id')
    ex_df = pd.DataFrame(
        {'id': grouped_by_id.groups.keys(), 'ts': grouped_by_id['t'].min(), 'tf': grouped_by_id['t'].max()})
    ex_df = ex_df.set_index(['id', 'ts', 'tf']).join(
        traj_df.rename(columns={'t': 'ts', 'x': 'xs', 'y': 'ys'}).set_index(['id', 'ts']))
    ex_df = ex_df.join(
        traj_df.rename(columns={'t': 'tf', 'x': 'xf', 'y': 'yf'}).set_index(['id', 'tf']).drop(columns='cat'))
    ex_df.reset_index(inplace=True)
    ex_df['interval'] = ex_df[['ts', 'tf']].apply(lambda x: pd.Interval(x['ts'], x['tf']), axis=1)

    print('Remove duplicate trajs...')
    ex_duplicated = ex_df.duplicated(subset=['ts', 'tf', 'xs', 'xf', 'ys', 'yf'])
    ex_df = ex_df.drop((ex_df.loc[ex_duplicated]).index)
    traj_df = traj_df.drop((traj_df[~(traj_df['id'].isin(ex_df['id']))]).index)
    print(f"Removed {ex_duplicated.sum()} trajectories\n")

    print('Unite broken trajs...')

    def close_ids(id, dt=pd.Timedelta(CLOSE_TIME, "s"), dl=CLOSE_DIST):
        cat = ex_df.loc[ex_df.id == id, 'cat'].values[0]
        tf = ex_df.loc[ex_df.id == id, 'tf'].values[0]
        xf = ex_df.loc[ex_df.id == id, 'xf'].values[0]
        yf = ex_df.loc[ex_df.id == id, 'yf'].values[0]

        start_close_t = ex_df.loc[(ex_df.cat == cat) & (pd.Timedelta(0, "s") < ex_df.ts - tf) & (ex_df.ts - tf < dt)]
        start_close_p = start_close_t.groupby('id').filter(lambda x: dl > abs(x['xs'].values[0] - xf))
        start_close_p = start_close_p.groupby('id').filter(lambda x: dl > abs(x['ys'].values[0] - yf))
        return start_close_p.id.values

    broken_df = pd.DataFrame({'id': ex_df['id']})
    broken_df['other_id'] = broken_df.apply(lambda x: close_ids(x['id']), axis=1)
    broken_df.set_index('id', inplace=True)

    broken_df = broken_df[broken_df['other_id'].map(lambda x: len(x)) > 0]
    print(f'Reunited {broken_df.shape[0]} trajectories\n')

    # Unite trajs
    def unite_traj(id):
        if id in broken_df.index:
            other = broken_df.loc[id, 'other_id']
            if other.size > 0:
                other = other[0]
                unite_traj(other)
                traj_df.loc[traj_df.id == other, 'id'] = id
                broken_df.drop(id)

    broken_df.reset_index().apply(lambda x: unite_traj(x.id), axis=1)

    print('Recalculating the extrema...')
    # Recalculate extrema
    grouped_by_id = traj_df.groupby('id')
    ex_df = pd.DataFrame(
        {'id': grouped_by_id.groups.keys(), 'ts': grouped_by_id['t'].min(), 'tf': grouped_by_id['t'].max()})
    ex_df = ex_df.set_index(['id', 'ts', 'tf']).join(
        traj_df.rename(columns={'t': 'ts', 'x': 'xs', 'y': 'ys'}).set_index(['id', 'ts']))
    ex_df = ex_df.join(
        traj_df.rename(columns={'t': 'tf', 'x': 'xf', 'y': 'yf'}).set_index(['id', 'tf']).drop(columns='cat'))
    ex_df.reset_index(inplace=True)
    ex_df['interval'] = ex_df[['ts', 'tf']].apply(lambda x: pd.Interval(x['ts'], x['tf']), axis=1)

    print('Removing too few points trajectories...')
    short_trajs = traj_df.groupby('id').filter(lambda x: len(x) <= MIN_POINTS)
    traj_df = traj_df.groupby('id').filter(lambda x: len(x) > MIN_POINTS)
    ex_df = ex_df.drop(ex_df[~ex_df['id'].isin(traj_df['id'])].index)
    print(f"Removed {short_trajs.id.unique().shape[0]} trajectories\n")

    print('Removing short trajectories...')
    static_trajs = ((abs(ex_df['xs'] - ex_df['xf']) < MIN_DIST) & (abs(ex_df['ys'] - ex_df['yf']) < MIN_DIST))
    print(f'Removed {ex_df.loc[static_trajs].shape[0]} trajectories\n')
    ex_df = ex_df.drop(ex_df[static_trajs].index)
    traj_df = traj_df.drop((traj_df[~(traj_df['id'].isin(ex_df['id']))]).index)

    print('Finding episodes...\n')

    def get_other_traj_ids(id):
        return \
            ex_df.loc[ex_df.set_index('interval').index.overlaps(ex_df.loc[ex_df['id'] == id, 'interval'].values[0])][
                'id'].values

    ep_df = pd.DataFrame({'id': ex_df.loc[ex_df.cat == 0, 'id']})
    ep_df['other_id'] = ep_df.apply(lambda x: get_other_traj_ids(x['id']), axis=1)

    print('Resampling...')

    # noinspection PyShadowingNames
    def resample(df):
        ndf = df.resample(FREQ_SAMPLE).asfreq()
        udf = pd.concat([df, ndf]).sort_index()
        return udf.interpolate(method='time').loc[ndf.index]

    grouped_by_id = traj_df[['t', 'id', 'cat', 'x', 'y']].set_index('t').groupby('id')
    traj_df = grouped_by_id.apply(lambda x: resample(x))
    traj_df = traj_df[['cat', 'x', 'y']]
    traj_df = traj_df.dropna()
    traj_df = traj_df.reset_index().drop_duplicates(['t', 'id'])
    times = traj_df.set_index('t').sort_index().index.unique()
    traj_df['it'] = traj_df.apply(lambda x: times.get_loc(x['t']), axis=1)
    traj_df = traj_df.drop('t', axis=1)
    traj_df = traj_df.rename(columns={'it': 't'})

    print('Calculating velocity and acceleration...\n')
    traj_df = traj_df.sort_values(by='t')
    traj_df['vx'] = traj_df.groupby('id')['x'].diff() * STEP_PER_SECOND
    traj_df['vy'] = traj_df.groupby('id')['y'].diff() * STEP_PER_SECOND
    traj_df['ax'] = traj_df.groupby('id')['vx'].diff() * STEP_PER_SECOND
    traj_df['ay'] = traj_df.groupby('id')['vy'].diff() * STEP_PER_SECOND
    traj_df = traj_df.dropna()

    print('Generating dataframes...')

    # Agent df
    agent_df = pd.DataFrame(
        {'id': traj_df.groupby('id').groups.keys(), 'ts': traj_df.groupby('id')['t'].min(),
         'tf': traj_df.groupby('id')['t'].max()})
    agent_df = agent_df.set_index(['id', 'ts', 'tf']).join(
        traj_df.rename(columns={'t': 'ts', 'x': 'xs', 'y': 'ys'}).set_index(['id', 'ts']).drop(
            columns=['vx', 'vy', 'ax', 'ay']))
    agent_df = agent_df.join(
        traj_df.rename(columns={'t': 'tf', 'x': 'xf', 'y': 'yf'}).set_index(['id', 'tf']).drop(
            columns=['vx', 'vy', 'ax', 'cat', 'ay']))
    agent_df.reset_index(inplace=True)

    # Simplified df
    car_df = traj_df.loc[traj_df.cat == 2, ['t', 'x', 'y', 'vx', 'ax']]
    upper_car = car_df.loc[(car_df.y > 0) & (car_df.x < rg.HIGH_X_INTERSECTION)].groupby('t').max('x')
    lower_car = car_df.loc[(car_df.y < 0) & (car_df.x > rg.LOW_X_INTERSECTION)].groupby('t').min('x')
    default_upper = DEFAULT_UPPER
    default_lower = DEFAULT_LOWER
    np_cars = np.array([[upper_car.loc[t, ['x', 'y', 'vx', 'ax']].values if t in upper_car.index else default_upper,
                         lower_car.loc[t, ['x', 'y', 'vx', 'ax']].values if t in lower_car.index else default_lower]
                        for t in range(0, traj_df.t.max())])

    # Only relevant episodes
    crossing_ped_df = agent_df.loc[agent_df.cat == 0].groupby('id').filter(
        lambda x: (((x.ys > rg.HIGH_Y_INTERSECTION) & (x.yf < rg.LOW_Y_INTERSECTION)) | (
                (x.ys < rg.LOW_Y_INTERSECTION) & (x.yf > rg.HIGH_Y_INTERSECTION)))).reset_index()

    # Create an ep_dict for the original df

    print('Saving results to disk...')
    np.save('Elaborated/np_cars.npy', np_cars)
    traj_df.to_pickle('Elaborated/traj_df.pkl')
    agent_df.to_pickle('Elaborated/agent_df.pkl')
    crossing_ped_df.to_pickle('Elaborated/crossing_ped_df.pkl')


if __name__ == '__main__':
    odf = pd.read_csv('original_df.csv', sep=';')
    df = odf[['ID', 'Tid', 'X', 'Y', 'Typ']]
    df.columns = ['id', 't', 'x', 'y', 'cat']

    df['t'] = pd.to_datetime(df['t'])
    df['t'] = df['t'].dt.tz_localize(None)

    clean_data(df)
