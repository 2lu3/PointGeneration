"""

Cyro電子顕微鏡で撮影したデータをするソフトウェア｡
主に､画像上のタンパク質の座標を抽出し､データベースと比較するときに使う｡
下の3つの機能がある｡

* 基準点のxyz座標から､指定されたパターンの点を生成する
* 生成された点ごとに､y軸を中心として始点からの距離に応じて回転させる
* 基準点の並びを直線と考え､y軸と並行に向けるためのオイラー角を計算する

Todo:
    * 回転角度/Åを自動で計算させる
    * docstringを充実させる

"""
import PySimpleGUI as sg
from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import math
import os
from scipy.spatial.transform import Rotation as Rot
from scipy import interpolate


class Calculator:
    """演算とファイル入出力を担当するclass"""

    def __init__(self):
        # 座標ファイルのPath
        self.input_file_path: str = ""
        # ファイルから読み込んだ基準点
        self.df_base_coordinates: Optional[pd.DataFrame] = None
        # 生成された点
        self.df_generated_coordinates: Optional[pd.DataFrame] = None
        # 生成されたオイラー角
        self.df_euler_degree: Optional[pd.DataFrame] = None
        # y軸並行にするためのRot
        self.rot2y: Optional[Rot] = None

    def load_text_file(self, path: str):
        self.input_file_path = path
        coordinates = []
        with open(path) as f:
            print("opened file: " + self.input_file_path)
            for line in f:
                coordinates.append(list(map(float, line.split())))
        print("reading file is completed")
        self.df_base_coordinates = pd.DataFrame(coordinates, columns=["x", "y", "z"])
        self.df_generated_coordinates = None
        print("converting read data is completed")

    def _save_df(self, df: pd.DataFrame, file_name: str):
        print("writing into " + file_name)
        with open(file_name, "w") as f:
            for _, row in df.iterrows():
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n")
        print("writing completed")

    def _add_suffix2path(self, file_path, suffix):
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path.split(".", 1)[0])
        ext = os.path.splitext(file_path)[1]
        return dir_name + os.sep + base_name + suffix + ext

    def save_positions(self):
        if self.df_generated_coordinates is not None:
            position_file_path = self._add_suffix2path(
                self.input_file_path, "_position"
            )
            self._save_df(self.df_generated_coordinates, position_file_path)

    def save_rotations(self):
        if self.df_euler_degree is not None:
            rotation_file_path = self._add_suffix2path(
                self.input_file_path, "_rotation"
            )
            self._save_df(self.df_euler_degree, rotation_file_path)

    def plot(self, ax):
        assert self.df_base_coordinates is not None
        ax.clear()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.scatter3D(self.df_base_coordinates["x"].values, self.df_base_coordinates["y"].values, self.df_base_coordinates["z"].values, color="r")  # type: ignore

        # グラフに表示するデータ。軸のスケールを揃えるのに使う
        df_all = self.df_base_coordinates.copy(deep=True)

        # 点が生成されている場合
        if self.df_generated_coordinates is not None:
            ax.scatter3D(self.df_generated_coordinates["x"].values, self.df_generated_coordinates["y"].values, self.df_generated_coordinates["z"].values, color="b", alpha=0.2)  # type: ignore
            ax.legend(["Original", "Generated"])

            df_all = pd.concat([df_all, self.df_generated_coordinates])

        else:
            ax.legend(["Original"])

        # 角軸のスケールを揃える
        x_max, y_max, z_max = df_all.max()
        x_min, y_min, z_min = df_all.min()
        max_range = max([x_max - x_min, y_max - y_min, z_max - z_min]) / 2

        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2

        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)

    def calculate_rot2y(self):
        """self.df_base_coordinatesの直線をy軸に並行にするRotを返す

        self.df_base_coordinatesには､複数の基準点が入っている｡
        基準点を直線と考え､y軸に並行な向きにするためにどう回転させるかを計算する｡

        raises:
            AssertionError: self.df_base_coordinatesがNoneの場合
        """
        assert self.df_base_coordinates is not None

        df_arrows: pd.DataFrame = self.df_base_coordinates.copy(deep=True)

        # 点同士を結ぶ直線のベクトルを計算する
        df_arrows = df_arrows.diff()
        df_arrows.drop(0, inplace=True)

        # ベクトルのx,y,z成分をそれぞれ抽出
        x: np.ndarray = df_arrows["x"].values  # type: ignore
        y: np.ndarray = df_arrows["y"].values  # type: ignore
        z: np.ndarray = df_arrows["z"].values  # type: ignore

        x = x.mean()
        y = y.mean()
        z = z.mean()

        # (x,y,z)の順番に回転角を計算
        radians = [
            -math.atan2(z, y),
            0,  # y軸に向けるので0
            math.atan2(x, math.sqrt(y ** 2 + z ** 2)),
        ]

        # 回転軸を設定
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        # 回転ベクトル
        vectors = [radians[i] * axes[i] for i in range(len(radians))]

        # Rotオブジェクト
        rots = [Rot.from_rotvec(vec) for vec in vectors]

        # Rotオブジェクト(x,y,zの順番に掛け合わせている)
        self.rot2y = rots[2] * rots[1] * rots[0]

        # 計算結果を保存
        rot_x, rot_y, rot_z = self.rot2y.as_euler('zxz',degrees=True)
        self.df_euler_degree = pd.DataFrame(
            [[math.degrees(rot_x), math.degrees(rot_y), math.degrees(rot_z)]],
            columns=["x", "y", "z"],
        )

    def _calc_spline(self):
        assert self.df_base_coordinates is not None
        base_x: np.ndarray = self.df_base_coordinates["x"].values  # type: ignore
        base_y: np.ndarray = self.df_base_coordinates["y"].values  # type: ignore
        base_z: np.ndarray = self.df_base_coordinates["z"].values  # type: ignore

        spline_tck, _ = interpolate.splprep([base_x, base_y, base_z])  # type: ignore

        # スプライン曲線の長さを計算する
        t = np.linspace(0, 1, 1000)

        x, y, z = interpolate.splev(t, spline_tck)
        x = np.diff(x)
        y = np.diff(y)
        z = np.diff(z)

        spline_length = np.sum(np.sqrt(x ** 2 + y ** 2 + z ** 2))
        return spline_tck, spline_length

    def _calculate_pitch_rotation(
        self,
        t: np.ndarray,
        spline_length: float,
        angstrom_per_pixel: float,
        pitch_degree: float,
        pitch_nm: float,
    ):
        try:
            pitch_radian_per_angstrom = math.radians(pitch_degree) / (pitch_nm * 10)
        except OverflowError:
            return
        except ZeroDivisionError:
            return

        # 始点からの距離に対して, pitchを適用する

        # 生成点ごとの回転角を計算する
        radians = t * spline_length * angstrom_per_pixel * pitch_radian_per_angstrom

        # y軸を中心としてradianだけ回転させるRotオブジェクト
        rot_y_axis: Rot = Rot.from_rotvec(
            np.array(
                [np.zeros(radians.shape[0]), radians, np.zeros(radians.shape[0])]
            ).T
        )

        assert self.rot2y is not None
        # 全体のRotオブジェクト
        rot_whole = rot_y_axis * self.rot2y

        # 全体の回転ベクトル
        rot_whole_vectors = rot_whole.as_euler('zxz', degrees=True)  # type: ignore

        self.df_euler_degree = pd.DataFrame()
        self.df_euler_degree["x"] = rot_whole_vectors[:, 0]
        self.df_euler_degree["y"] = rot_whole_vectors[:, 1]
        self.df_euler_degree["z"] = rot_whole_vectors[:, 2]

    def generate_along_line(
        self,
        angstrom_per_pixel: float,
        distance_nm: float,
        pitch_degree: Optional[float] = None,
        pitch_nm: Optional[float] = None,
    ):
        spline_tck, spline_length = self._calc_spline()

        distance_angstrom = distance_nm * 10

        try:
            if distance_angstrom == 0:
                raise OverflowError
            num = int(spline_length * angstrom_per_pixel / distance_angstrom)
        except OverflowError:
            # 値をユーザーが入力中
            self.reset_calculation()
            return

        if num == 0:
            return

        # 媒介変数t. 0<=t<=1でスプライン曲線の始点から終点までをカバーできる
        t = np.linspace(0, 1, num)
        x, y, z = interpolate.splev(t, spline_tck)
        self.df_generated_coordinates = pd.DataFrame()
        self.df_generated_coordinates["x"] = x
        self.df_generated_coordinates["y"] = y
        self.df_generated_coordinates["z"] = z

        if pitch_degree is not None and pitch_nm is not None:
            self._calculate_pitch_rotation(
                t, spline_length, angstrom_per_pixel, pitch_degree, pitch_nm
            )
        else:
            self.df_euler_degree = None

    def generate_in_column(
        self,
        angstrom_per_pixel: float,
        radius_nm: float,
        point_number: int,
        pitch_degree: Optional[float] = None,
        pitch_nm: Optional[float] = None,
    ):
        """スプライン曲線を中心とした円柱の中で、ランダムに点を生成する


        数学的な計算方法:
            目標: スプライン曲線の方向と垂直な平面でthetaとrを指定したい

            (以下の処理は、生成された点ごとにそれぞれ別の原点を用意する)
            スプライン曲線上に生成された点をある原点とし、
            生データのデカルト座標系を平行移動させた座標系で議論する

            スプライン曲線を微分し、スプライン曲線の方向を示す単位ベクトル(dx, dy, dz)を計算する
            点(dx, dy, 0)と原点を結ぶ直線と垂直で、かつ原点を通る直線をxy平面上で求める
            その直線の単位ベクトルを、vertical_vector とする

            xy平面上のvertical_vectorを3次元座標系に変換する(z=0を付け加えるだけ)
            vertical_vector を回転ベクトル(dx * theta, dy * theta, dz * theta)で回転させ、r倍すると、求めたい座標を得られる

            theta,rを計算する平面とその軸は(dx, dy, dz)に依存するが、r,thetaともに乱数で、ランダムな座標が得られれば良いので、問題ないと考えた
        """

        try:
            radius_pixel = radius_nm * 10 / angstrom_per_pixel
        except (OverflowError, ZeroDivisionError):
            # 値をユーザーが入力中
            self.reset_calculation()
            return

        if point_number <= 0:
            return

        spline_tck, spline_length = self._calc_spline()

        theta = np.random.uniform(0, math.pi * 2, point_number)
        r = np.sqrt(np.random.uniform(0, radius_pixel ** 2, point_number))
        # 媒介変数t(0<=t<=1)でスプライン曲線上での位置を指定する
        t = np.linspace(0, 1, point_number)

        # スプライン曲線上でのある位置のベクトルを求めるために使う
        dt = 1 / point_number / 10

        x, y, z = interpolate.splev(t, spline_tck)
        x_plus_dx, y_plus_dy, z_plus_dz = interpolate.splev(t + dt, spline_tck)

        dx = x_plus_dx - x
        dy = y_plus_dy - y
        dz = z_plus_dz - z

        abs_delta = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        dx /= abs_delta
        dy /= abs_delta
        dz /= abs_delta

        # xy平面上の、ベクトル(dx, dy, dz)と垂直な単位ベクトル
        vertical_vector = np.array(
            [
                dy / np.sqrt(dx ** 2 + dy ** 2),
                -dx / np.sqrt(dx ** 2 + dy ** 2),
                np.zeros(point_number),
            ]
        ).T

        rot_vector = Rot.from_rotvec(np.array([dx * theta, dy * theta, dz * theta]).T)
        rotated_vector = rot_vector.apply(vertical_vector)

        self.df_generated_coordinates = pd.DataFrame(columns=["x", "y", "z"])
        self.df_generated_coordinates["x"] = rotated_vector[:, 0] * r + x
        self.df_generated_coordinates["y"] = rotated_vector[:, 1] * r + y
        self.df_generated_coordinates["z"] = rotated_vector[:, 2] * r + z

        if pitch_degree is not None and pitch_nm is not None:
            self._calculate_pitch_rotation(
                t, spline_length, angstrom_per_pixel, pitch_degree, pitch_nm
            )

    def reset_calculation(self):
        """ファイルをロードした時点まで戻る"""
        self.df_generated_coordinates = None
        self.df_euler_degree = None
        self.calculate_rot2y()


class GUI:
    """表示部分を担当するclass

    表示する画面のデザイン・管理を担当する

    """

    def __init__(self):
        self.calculator = Calculator()
        self.font = "Calibri"
        self.fontsize = {
            "title": 50,
            "default": 30,
            "detail": 15,
        }
        self.colors = {
            "default_button": ("#FFFFFF", "#283b5b"),
            "special_button": "royalblue",
            "special_title": "royalblue",
        }
        # 1~3のみ
        self.step: int = 1

        self.window = None
        # window の位置
        self.window_location = None

        # グラフ描画用
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection="3d")

        # ユーザーが入力したデータを保存する｡(Backボタン用)
        self.parameters = {
            "STEP1": {
                "-PATH POSITION FILE-": "",
                "-PATH MRC FILE-": "",
            },
            "STEP2": {
                "-ALONG LINE-": False,
                "-IN COLUMN-": False,
                "-ROTATION FOR Y AXIS-": False,
                "-ROTATION FOR ROTATING PROTAIN-": False,
            },
            "STEP3": {
                "-ANGSTROM PER PIXEL-": "",
                "-POINT DISTANCE-": "",
                "-MAX RADIUS-": "",
                "-POINT NUMBER-": "",
                "-PITCH DEGREE-": "",
                "-PITCH DISTANCE-": "",
            },
        }

    def _create_layout(self):
        def _layout_header():
            padding = 30
            return sg.Column(
                [
                    [
                        sg.Text(
                            "Choose File",
                            font=(self.font, self.fontsize["title"]),
                            pad=(padding, 0),
                            justification="center",
                            key="-STEP1 TITLE-",
                        ),
                        sg.Text(
                            "->",
                            font=(self.font, self.fontsize["default"]),
                            justification="center",
                        ),
                        sg.Text(
                            "Select type",
                            font=(self.font, self.fontsize["title"]),
                            pad=(padding, 0),
                            justification="center",
                            key="-STEP2 TITLE-",
                        ),
                        sg.Text(
                            "->",
                            font=(self.font, self.fontsize["default"]),
                            justification="center",
                        ),
                        sg.Text(
                            "Set parameter",
                            font=(self.font, self.fontsize["title"]),
                            pad=(padding, 0),
                            justification="center",
                            key="-STEP3 TITLE-",
                        ),
                    ]
                ],
                pad=((0, 0), (0, 50)),
            )

        def _layout_foot():
            padding_center = 30

            back_button = sg.Button(
                "Back",
                font=(self.font, self.fontsize["default"]),
                pad=(padding_center, 0),
                key="-BACK-",
            )

            next_button = sg.Button(
                "Next",
                font=(self.font, self.fontsize["default"]),
                tooltip="You can return to this display later",
                pad=(padding_center, 0),
                key="-NEXT-",
            )

            save_button = sg.Button(
                "Save",
                font=(self.font, self.fontsize["default"]),
                pad=(padding_center, 0),
                key="-SAVE-",
            )

            return sg.Column(
                [[back_button, next_button, save_button]], justification="center"
            )

        def _layout_step1():
            padding_center = 40
            padding_top = 60
            padding_bottom = 60
            input_text_size = 40

            position_title = sg.Text(
                "Position File",
                font=(self.font, self.fontsize["default"]),
                justification="center",
                expand_x=True,
            )
            position_file = [
                sg.InputText(
                    size=input_text_size,
                    font=(self.font, self.fontsize["detail"]),
                    key="-PATH POSITION FILE-",
                ),
                sg.FileBrowse(
                    "Open",
                    font=(self.font, self.fontsize["default"]),
                    file_types=(
                        ("Text files", "*.txt"),
                        ("All files", "*"),
                    ),
                ),
            ]
            column_position = sg.Column(
                [[position_title], position_file],
                pad=((0, padding_center), (padding_top, padding_bottom)),
            )

            mrc_title = sg.Text(
                "MRC File",
                font=(self.font, self.fontsize["default"]),
                justification="center",
                expand_x=True,
            )
            mrc_file = [
                sg.InputText(
                    size=input_text_size,
                    key="-PATH MRC FILE-",
                    font=(self.font, self.fontsize["detail"]),
                ),
                sg.FileBrowse(
                    "Open",
                    font=(self.font, self.fontsize["default"]),
                    file_types=(
                        ("MRC files", "*.mrc"),
                        ("All files", "*"),
                    ),
                ),
            ]
            column_mrc = sg.Column(
                [[mrc_title], mrc_file],
                pad=((padding_center, 0), (padding_top, padding_bottom)),
            )

            return sg.Column([[column_position, column_mrc]], justification="center")

        def _layout_step2():
            padding = 20
            line_button = sg.Button(
                "Along Line",
                tooltip="入力ファイルから読み取った基準点からスプライン曲線もどきを生成します。\n"
                + "その後、スプライン曲線上に指定の間隔で点を生成・出力できます。",
                font=(self.font, self.fontsize["default"]),
                pad=padding,
                key="-ALONG LINE-",
            )
            column_button = sg.Button(
                "In Column",
                font=(self.font, self.fontsize["default"]),
                tooltip="入力ファイルから読み取った基準点からスプライン曲線もどきを生成します。\n"
                + "その後、スプライン曲線を中心とした円柱を生成し、円柱内でランダムに点を生成・出力できます。",
                pad=padding,
                key="-IN COLUMN-",
            )
            rot2y_button = sg.Button(
                "Rotation for y axis",
                font=(self.font, self.fontsize["default"]),
                tooltip="入力ファイルから読み取った基準点を直線に近似し、\n" + "その直線をy軸に平行にできる回転ベクトルを出力します。",
                pad=padding,
                key="-ROTATION FOR Y AXIS-",
            )
            pitch_button = sg.Button(
                "Rotation for rotating protain",
                font=(self.font, self.fontsize["default"]),
                tooltip="入力ファイルから読み取った基準点からスプライン曲線もどきを生成します。\n"
                + "その後、スプライン曲線の周りでらせんを描く構造があると仮定し、\n"
                + "らせんをy軸に平行にできる回転ベクトルを出力します。\n"
                + "この回転ベクトルは、Rotation for y axisの回転ベクトルと合成されています。",
                pad=padding,
                key="-ROTATION FOR ROTATING PROTAIN-",
            )
            frame_generation = sg.Frame(
                "Position Generation",
                [[line_button], [column_button]],
                font=(self.font, self.fontsize["default"]),
                element_justification="center",
                pad=padding,
            )
            frame_calc = sg.Frame(
                "Rotation Calculation",
                [[rot2y_button], [pitch_button]],
                font=(self.font, self.fontsize["default"]),
                element_justification="center",
                pad=padding,
            )
            return sg.Column([[frame_generation, frame_calc]], justification="center")

        def _layout_step3():
            input_text_size = 10
            padding = 20
            column_name = sg.Column(
                [
                    [
                        sg.Text(
                            "Angstrom / pixel",
                            font=(self.font, self.fontsize["default"]),
                        ),
                    ],
                    [
                        sg.Text(
                            "Point Distance (nm)",
                            font=(self.font, self.fontsize["default"]),
                        ),
                    ],
                    [
                        sg.Text(
                            "Max Radius (nm)",
                            font=(self.font, self.fontsize["default"]),
                        )
                    ],
                    [
                        sg.Text(
                            "Point Number",
                            font=(self.font, self.fontsize["default"]),
                        ),
                    ],
                    [
                        sg.Text(
                            "Pitch Degree", font=(self.font, self.fontsize["default"])
                        ),
                    ],
                    [
                        sg.Text(
                            "Pitch Distance (nm)",
                            font=(self.font, self.fontsize["default"]),
                        ),
                    ],
                ],
            )
            column_input_text = sg.Column(
                [
                    [
                        sg.InputText(
                            key="-ANGSTROM PER PIXEL-",
                            size=input_text_size,
                            font=(self.font, self.fontsize["default"]),
                            enable_events=True,
                        ),
                    ],
                    [
                        sg.InputText(
                            key="-POINT DISTANCE-",
                            size=input_text_size,
                            font=(self.font, self.fontsize["default"]),
                            enable_events=True,
                        ),
                    ],
                    [
                        sg.InputText(
                            key="-MAX RADIUS-",
                            size=input_text_size,
                            font=(self.font, self.fontsize["default"]),
                            enable_events=True,
                        )
                    ],
                    [
                        sg.InputText(
                            key="-POINT NUMBER-",
                            size=input_text_size,
                            font=(self.font, self.fontsize["default"]),
                            enable_events=True,
                        ),
                    ],
                    [
                        sg.InputText(
                            key="-PITCH DEGREE-",
                            size=input_text_size,
                            font=(self.font, self.fontsize["default"]),
                            enable_events=True,
                        ),
                    ],
                    [
                        sg.InputText(
                            key="-PITCH DISTANCE-",
                            size=input_text_size,
                            font=(self.font, self.fontsize["default"]),
                            enable_events=True,
                        ),
                    ],
                ],
                pad=padding,
            )
            canvas_plot = sg.Canvas(key="-PLOT CANVAS-")

            return sg.Column(
                [[column_name, column_input_text, canvas_plot]], justification="center"
            )

        def _layout_main():
            if self.step == 1:
                return _layout_step1()
            elif self.step == 2:
                return _layout_step2()
            elif self.step == 3:
                return _layout_step3()
            else:
                raise ValueError("self.step is " + str(self.step))

        layout = [
            [_layout_header()],
            [_layout_main()],
            [_layout_foot()],
        ]

        if self.window is not None:
            self.window_location = self.window.CurrentLocation()
            self.window.close()
            self.window: Any = sg.Window(
                "Hikaru's Magic Software",
                layout,
                finalize=True,
                location=self.window_location,
            )
        else:
            self.window: Any = sg.Window(
                "Hikaru's Magic Software",
                layout,
                finalize=True,
            )

        # 状況に応じた調整
        self.window["-STEP" + str(self.step) + " TITLE-"].update(
            background_color=self.colors["special_title"]
        )

        if self.step == 1:
            self.window["-BACK-"].update(visible=False)
            self.window["-SAVE-"].update(visible=False)
            for key in self.parameters["STEP1"].keys():
                self.window[key].update(self.parameters["STEP1"][key])
        elif self.step == 2:
            self.window["-SAVE-"].update(visible=False)
            for key in self.parameters["STEP2"].keys():
                if self.parameters["STEP2"][key] is True:
                    self.window[key].update(button_color=self.colors["special_button"])
        elif self.step == 3:
            self.window["-NEXT-"].update(visible=False)

            # 保存されていた値を画面上に表示
            for key in self.parameters["STEP3"].keys():
                self.window[key].update(self.parameters["STEP3"][key])

            # オプションに合わせていらない入力ができないようにする
            if self.parameters["STEP2"]["-ALONG LINE-"] is False:
                self.window["-POINT DISTANCE-"].update(visible=False)
            if self.parameters["STEP2"]["-IN COLUMN-"] is False:
                self.window["-MAX RADIUS-"].update(visible=False)
                self.window["-POINT NUMBER-"].update(visible=False)
            if self.parameters["STEP2"]["-ROTATION FOR ROTATING PROTAIN-"] is False:
                self.window["-PITCH DEGREE-"].update(visible=False)
                self.window["-PITCH DISTANCE-"].update(visible=False)

            if (
                self.parameters["STEP2"]["-ALONG LINE-"] is False
                and self.parameters["STEP2"]["-IN COLUMN-"] is False
                and self.parameters["STEP2"]["-ROTATION FOR ROTATING PROTAIN-"] is False
            ):
                self.window["-ANGSTROM PER PIXEL-"].update(visible=False)

            # figure と canvas を紐付ける
            self.figure_canvas = FigureCanvasTkAgg(
                self.figure, self.window["-PLOT CANVAS-"].TKCanvas
            )
            self.figure_canvas.draw()
            self.figure_canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

            self._calculate_graph()
            self._plot()

        else:
            assert ValueError("self.step is " + str(self.step))

        return

    def _calculate_graph(self):
        """条件にしたがって、self.calculatorで演算を行う

        STEP2で入力したパラメーターと、STEP3で入力中のパラメーターを用いて計算する

        * 考慮するSTEP2のパラメーターと必要なSTEP3のパラメーター
        STEP2パラメーター
            * 必要なSTEP3のパラメーター
        -ALONG LINE-
            * -ANGSTROM PER PIXEL-
            * -POINT DISTANCE-
        -IN COLUMN-
            * -ANGSTROM PER PIXEL-
            * -POINT NUMBER-
            * -MAX RADIUS-
        -ROTATION FOR ROTATING PROTAIN-
            * -ANGSTROM PER PIXEL-
            * -PITCH DEGREE-
            * -PITCH DISTANCE-

        * 考慮しないSTEP2のパラメーター
        -ROTATION FOR Y AXIS-
            calculatorがファイルの読み込みを行った時点で、計算されているため

        Notes:
            -ROTATION FOR ROTATING PROTAIN-は-ALONG LINE-または-IN COLUMN-と共存必須
            画面が変わってもパラメーターを保持しているため、
            画面に表示されていなくても数値が入っている可能性がある

        """

        try:
            pitch_degree = float(self.parameters["STEP3"]["-PITCH DEGREE-"])
            pitch_distance = float(self.parameters["STEP3"]["-PITCH DISTANCE-"])
        except ValueError:
            # float に変換できない場合
            pitch_degree = None
            pitch_distance = None

        if self.parameters["STEP2"]["-ROTATION FOR ROTATING PROTAIN-"] is False:
            pitch_degree = None
            pitch_distance = None

        # angstrom_per_pixel は未選択以外どのパラメーターでも必要
        try:
            angstrom_per_pixel = float(self.parameters["STEP3"]["-ANGSTROM PER PIXEL-"])
        except ValueError:
            # float に変換できない場合
            # angstrom_per_pixel は必須情報
            self.calculator.reset_calculation()
            return

        if self.parameters["STEP2"]["-IN COLUMN-"] is True:
            try:
                point_number = int(self.parameters["STEP3"]["-POINT NUMBER-"])
                max_radius = float(self.parameters["STEP3"]["-MAX RADIUS-"])
            except ValueError:
                # 必須情報が足りない
                return
            self.calculator.generate_in_column(
                angstrom_per_pixel,
                max_radius,
                point_number,
                pitch_degree,
                pitch_distance,
            )
        elif self.parameters["STEP2"]["-ALONG LINE-"] is True:
            try:
                point_distance = float(self.parameters["STEP3"]["-POINT DISTANCE-"])
            except ValueError:
                # 必須条件が足りない
                return

            self.calculator.generate_along_line(
                angstrom_per_pixel,
                point_distance,
                pitch_distance,
                pitch_distance,
            )
        else:
            # 無選択
            self.calculator.reset_calculation()
            return

    def _plot(self):
        self.calculator.plot(self.ax)
        self.figure_canvas.draw()

    def run(self):
        def _save_parameters():
            """self.parametersに保存する"""
            for key in self.parameters["STEP" + str(self.step)].keys():
                self.parameters["STEP" + str(self.step)][key] = values[key]

        def _on_click_back_button():
            if self.step == 2:
                # 実装するべきことなし
                pass
            elif self.step == 3:
                _save_parameters()
            else:
                # 実行されることなし
                raise RuntimeError(
                    "GUI._on_click_back_button() called when self.step == 1"
                )

            self.step -= 1
            self._create_layout()

        def _on_click_next_button():
            """NEXT ボタンが押されたときの処理を行う"""
            if self.step == 1:
                _save_parameters()
                try:
                    self.calculator.load_text_file(values["-PATH POSITION FILE-"])
                    self.calculator.calculate_rot2y()
                except FileNotFoundError as e:
                    sg.popup_error("Sorry. We could not find the file you entered.")
                    print(e)
                    return
            elif self.step == 2:
                pass
            else:
                # 実行されることなし
                raise RuntimeError(
                    "GUI._on_click_next_button() called when self.step == 3"
                )

            self.step += 1
            self._create_layout()
            print("re-created layout")

        def _on_click_save_button():
            try:
                flag_saved = False
                if (
                    self.parameters["STEP2"]["-ALONG LINE-"] is True
                    or self.parameters["STEP2"]["-IN COLUMN-"] is True
                ):
                    self.calculator.save_positions()
                    flag_saved = True
                if (
                    self.parameters["STEP2"]["-ROTATION FOR Y AXIS-"] is True
                    or self.parameters["STEP2"]["-ROTATION FOR ROTATING PROTAIN-"]
                    is True
                ):
                    flag_saved = True
                    self.calculator.save_rotations()
                if flag_saved is False:
                    sg.popup("Nothing saved according to the options you selected")
            except AssertionError:
                sg.popup_error("Internal Error: self.calculator.save() failed")

        def _update_step2_button(key: str, toggle: bool):
            self.parameters["STEP2"][key] = toggle
            if toggle is True:
                self.window[key].update(button_color=self.colors["special_button"])
            else:
                self.window[key].update(button_color=self.colors["default_button"])

        self._create_layout()

        while True:
            event, values = self.window.read()

            if event is None:
                break
            elif event == "-BACK-":
                _on_click_back_button()
                continue
            elif event == "-NEXT-":
                _on_click_next_button()
                continue
            elif event == "-SAVE-":
                _on_click_save_button()
                continue

            if self.step == 1:
                # 実装するべきことなし
                pass
            elif self.step == 2:
                # TrueをFalseに､FalseをTrueに
                self.parameters["STEP2"][event] = (
                    self.parameters["STEP2"][event] is False
                )

                # 変更したボタンの色を変更する
                if self.parameters["STEP2"][event] is True:
                    self.window[event].update(
                        button_color=self.colors["special_button"]
                    )
                else:
                    self.window[event].update(
                        button_color=self.colors["default_button"]
                    )

                # 依存関係に対応
                if self.parameters["STEP2"][event] is True:
                    if event == "-ALONG LINE-":
                        # -IN COLUMN- と共存不可
                        _update_step2_button("-IN COLUMN-", False)
                    elif event == "-IN COLUMN-":
                        # -ALONG LINE- と共存不可
                        _update_step2_button("-ALONG LINE-", False)
                    elif event == "-ROTATION FOR ROTATING PROTAIN-":
                        # -IN COLUMN- または -ALONG LINE- を選択する必要がある
                        if (
                            self.parameters["STEP2"]["-ALONG LINE-"] is False
                            and self.parameters["STEP2"]["-IN COLUMN-"] is False
                        ):
                            # どちらでも良いが、デフォルトで -ALONG LINE- を自動で選択するように
                            _update_step2_button("-ALONG LINE-", True)
                        # -ROTATION FOR Y AXIS- を選択する必要がある
                        _update_step2_button("-ROTATION FOR Y AXIS-", True)
                else:
                    if (
                        event == "-ROTATION FOR Y AXIS-"
                        or event == "-IN COLUMN-"
                        or event == "-ALONG LINE-"
                    ):
                        # -ROTATION FOR ROTATING PROTAIN- を解除
                        _update_step2_button("-ROTATION FOR ROTATING PROTAIN-", False)
            elif self.step == 3:
                _save_parameters()
                self._calculate_graph()
                self._plot()
                pass

        self.window.close()


def main():
    gui = GUI()
    gui.run()


def tester():
    df = pd.DataFrame()
    for _ in range(1000):
        calculator = Calculator()
        calculator.load_text_file("./data.txt")
        calculator.calculate_rot2y()

        calculator.generate_in_column(1, 10, 1, 1)


if __name__ == "__main__":
    main()
    # tester()
